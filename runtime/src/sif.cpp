// SIF: the EE<->IOP link, emulated at the command-packet level.
//
// The EE libraries talk to IOP modules with SIF command packets sent by DMA
// (SifSetDma). We parse those packets and answer as the IOP would: replies
// are written into the EE's receive buffer and delivered through the game's
// own SIF0 DMA interrupt handler, one packet per safe point.
#include "rt_internal.h"
#include "hw.h"
#include "sif.h"
#include "audio.h"
#include <deque>
#include <map>
#include <vector>

enum : uint32_t {
    SIF_CMD_CHANGE_SADDR = 0x80000000, SIF_CMD_SET_SREG = 0x80000001, SIF_CMD_INIT_CMD = 0x80000002,
    SIF_CMD_RESET = 0x80000003, SIF_CMD_RPC_END = 0x80000008, SIF_CMD_RPC_BIND = 0x80000009,
    SIF_CMD_RPC_CALL = 0x8000000A, SIF_CMD_RPC_RDATA = 0x8000000C,
};

// IOP-side addresses we hand out (IOP RAM is just a scratch area here)
static const uint32_t IOP_CMD_BUF = 0x0001F000;  // EE writes command packets here
static uint32_t g_iop_alloc = 0x00100000;        // per-server receive buffers

static uint32_t g_ee_pktbuf;                     // EE receive buffer (CHANGE_SADDR)
static std::map<uint32_t, uint32_t> g_sreg;      // SIF software registers
static std::deque<std::vector<uint8_t>> g_pending;
static bool g_iop_rebooting;

struct Server { uint32_t sid; uint32_t buff; SifRpcHandler fn; const char *name; };
static std::map<uint32_t, Server> g_servers;      // by sid
static std::map<uint32_t, uint32_t> g_server_by_ptr;

void iop_register_unknown(uint32_t sid);
static uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static void wr32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }

void sif_register_server(uint32_t sid, const char *name, SifRpcHandler fn) {
    Server s{sid, 0, fn, name};
    s.buff = g_iop_alloc; g_iop_alloc += 0x10000;
    g_servers[sid] = s;
    g_server_by_ptr[0x00200000 - 0x100 - (uint32_t)g_servers.size() * 0x40] = sid;
}

static uint32_t server_ptr(uint32_t sid) {
    for (auto &kv : g_server_by_ptr) if (kv.second == sid) return kv.first;
    uint32_t p = 0x00180000 + (uint32_t)g_server_by_ptr.size() * 0x40;
    g_server_by_ptr[p] = sid;
    return p;
}

static void queue_reply(std::vector<uint8_t> pkt) { g_pending.push_back(std::move(pkt)); }

static std::vector<uint8_t> make_rend(const uint8_t *req, uint32_t cid, uint32_t server, uint32_t buff, uint32_t cbuff) {
    std::vector<uint8_t> p(48, 0);
    wr32(&p[0], 48);                      // psize = 48, dsize = 0
    wr32(&p[8], SIF_CMD_RPC_END);
    memcpy(&p[16], req + 16, 16);         // rec_id, pkt_addr, rpc_id, client
    wr32(&p[32], cid);
    wr32(&p[36], server);
    wr32(&p[40], buff);
    wr32(&p[44], cbuff);
    return p;
}

static const char *server_name(uint32_t sid) {
    auto it = g_servers.find(sid);
    return it == g_servers.end() ? "unknown" : it->second.name;
}

static void handle_packet(const uint8_t *p, uint32_t size) {
    uint32_t psize = rd32(p) & 0xFF, cid = rd32(p + 8);
    (void)psize;
    switch (cid) {
    case SIF_CMD_CHANGE_SADDR:
        g_ee_pktbuf = rd32(p + 16);
        rt_log(LOG_INFO, "SIF: EE receive buffer at %08X", g_ee_pktbuf);
        break;
    case SIF_CMD_SET_SREG:
        g_sreg[rd32(p + 16)] = rd32(p + 20);
        break;
    case SIF_CMD_INIT_CMD:
        rt_log(LOG_INFO, "SIF: init cmd (opt=%08X)", rd32(p + 12));
        if (rd32(p + 12) == 0) g_ee_pktbuf = rd32(p + 16);
        else {
            // RPC init: the IOP acknowledges by setting EE software register 0
            std::vector<uint8_t> r(24, 0);
            wr32(&r[0], 24);
            wr32(&r[8], SIF_CMD_SET_SREG);
            wr32(&r[16], 0);
            wr32(&r[20], 1);
            queue_reply(std::move(r));
        }
        break;
    case SIF_CMD_RESET:
        rt_log(LOG_INFO, "SIF: IOP reset requested (%s)", (const char *)(p + 24));
        g_iop_rebooting = true;   // reports booted on the next status poll
        break;
    case SIF_CMD_RPC_BIND: {
        uint32_t sid = rd32(p + 32);
        if (!g_servers.count(sid)) iop_register_unknown(sid);
        auto it = g_servers.find(sid);
        uint32_t buff = it != g_servers.end() ? it->second.buff : 0;
        uint32_t srv = (it != g_servers.end()) ? server_ptr(sid) : 0;
        if (rt_once(0xC00000000ull | sid))
            rt_log(it != g_servers.end() ? LOG_INFO : LOG_WARN, "SIF: bind %08X (%s)%s", sid, server_name(sid),
                   srv ? "" : " -> no such server, client will retry");
        queue_reply(make_rend(p, SIF_CMD_RPC_BIND, srv, buff, buff));
        break;
    }
    case SIF_CMD_RPC_CALL: {
        uint32_t fno = rd32(p + 32), ssize = rd32(p + 36), recv = rd32(p + 40), rsize = rd32(p + 44);
        uint32_t srv = rd32(p + 52);
        auto sp = g_server_by_ptr.find(srv);
        uint32_t sid = sp != g_server_by_ptr.end() ? sp->second : 0;
        auto it = g_servers.find(sid);
        SifRpcCall call{sid, fno, it != g_servers.end() ? it->second.buff : 0, ssize, recv, rsize};
        if (it != g_servers.end() && it->second.fn) {
            it->second.fn(call);
        } else if (rt_once(0xD00000000ull | ((uint64_t)sid << 8) | (fno & 0xFF))) {
            rt_log(LOG_WARN, "SIF: call %08X (%s) fn %u send %X recv %X: no handler", sid, server_name(sid), fno, ssize, rsize);
        }
        queue_reply(make_rend(p, SIF_CMD_RPC_CALL, srv, call.buff, call.buff));
        break;
    }
    default:
        if (rt_once(0xE00000000ull | cid)) rt_log(LOG_WARN, "SIF: unhandled command %08X size %u", cid, size);
    }
}

uint32_t sif_set_dma(uint32_t desc, uint32_t count) {
    // SifDmaTransfer_t { void *src; void *dest; int size; int attr; }
    for (uint32_t i = 0; i < count; i++) {
        uint32_t d = desc + i * 16;
        uint32_t src = LW(d), dst = LW(d + 4) & 0x1FFFFF, size = LW(d + 8);
        std::vector<uint8_t> buf(size);
        for (uint32_t k = 0; k < size; k++) buf[k] = LB(src + k);
        if (uint8_t *pool = iop_audio_ram(LW(d + 4), size)) {   // sound driver wave memory
            std::lock_guard<std::mutex> lk(g_audio_mx);
            memcpy(pool, buf.data(), size);
            continue;
        }
        memcpy(g_iop_ram + dst, buf.data(), std::min<size_t>(size, sizeof g_iop_ram - dst));
        if (dst == IOP_CMD_BUF && size >= 16) handle_packet(buf.data(), size);
    }
    static uint32_t id;
    return ++id;
}

// Deliver one queued IOP->EE packet through the EE's SIF0 DMA handler.
bool sif_deliver_one() {
    if (g_pending.empty() || !g_ee_pktbuf) return false;
    auto pkt = std::move(g_pending.front());
    g_pending.pop_front();
    for (size_t k = 0; k < pkt.size(); k++) SB(g_ee_pktbuf + (uint32_t)k, pkt[k]);
    hw_set_reg(D_STAT, hw_reg(D_STAT) | (1u << 5));
    kernel_dmac_event(5);
    return true;
}

bool sif_has_pending() { return !g_pending.empty(); }

void sif_write_ee(uint32_t ee_addr, const void *data, uint32_t size) {
    const uint8_t *p = (const uint8_t *)data;
    for (uint32_t k = 0; k < size; k++) SB(ee_addr + k, p[k]);
}

uint32_t sif_get_reg(uint32_t r) {
    uint32_t v;
    switch (r) {
    case 1: v = hw_reg(SIF_MSCOM); break;
    case 2: v = hw_reg(SIF_SMCOM); break;
    case 3: v = hw_reg(SIF_MSFLG); break;
    case 4:
        if (g_iop_rebooting) {  // IOP finished rebooting: SIFINIT | CMDINIT | BOOTEND
            g_iop_rebooting = false;
            hw_set_reg(SIF_SMFLG, hw_reg(SIF_SMFLG) | 0x70000u);
            rt_log(LOG_INFO, "SIF: IOP reboot complete");
        }
        v = hw_reg(SIF_SMFLG); break;
    case 0x80000000u: v = IOP_CMD_BUF; break;          // SUBADDR: IOP command buffer
    default: { auto it = g_sreg.find(r); v = it == g_sreg.end() ? 0 : it->second; }
    }
    if (rt_once(0x800000000ull | r)) rt_log(LOG_INFO, "SifGetReg(%08X) -> %08X", r, v);
    return v;
}

void sif_set_reg(uint32_t r, uint32_t v) {
    switch (r) {
    case 1: hw_set_reg(SIF_MSCOM, v); break;
    case 3: hw_set_reg(SIF_MSFLG, hw_reg(SIF_MSFLG) | v); break;
    case 4: hw_set_reg(SIF_SMFLG, hw_reg(SIF_SMFLG) & ~v); break;
    default: g_sreg[r] = v;
    }
    if (rt_once(0x900000000ull | r)) rt_log(LOG_INFO, "SifSetReg(%08X, %08X)", r, v);
}

void sif_flag_write(uint32_t, uint32_t) {}

void iop_modules_init();  // iop_modules.cpp
void sif_init() { iop_modules_init(); }
