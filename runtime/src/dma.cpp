// EE DMA controller (instant transfers) and the VIF0/VIF1 command processors.
#include "rt_internal.h"
#include "hw.h"
#include "gs.h"
#include "vu.h"
#include <algorithm>
#include <string>
#include <vector>

void kernel_intc_event(int cause);   // kernel.cpp

static uint8_t *dma_ptr(uint32_t addr, bool spr) {
    if (spr || (addr & 0x80000000u)) return g_spr + (addr & 0x3FF0);
    return g_ram + (addr & (RAM_SIZE - 16));
}

// ================================================================ PATH3 gating
// In-game the GIF channel carries one frame's texture uploads (CLUT + image + TEXFLUSH,
// one EOP-terminated packet per texture) while the VIF1 chain draws. VIF1 brackets each
// object with MSKPATH3 0 / MSKPATH3 1, letting exactly one packet through before its draw
// (GIF arbitration only switches paths at packet boundaries). So PATH3 data is queued and
// released packet by packet: one on unmask, everything while left unmasked.
static std::vector<uint8_t> g_p3q;
static size_t g_p3pos;
static bool g_p3_masked;

static size_t p3_packet_end(size_t pos) {   // byte offset just past the next EOP packet
    while (pos + 16 <= g_p3q.size()) {
        uint64_t lo; memcpy(&lo, &g_p3q[pos], 8);
        uint32_t nloop = (uint32_t)(lo & 0x7FFF), flg = (uint32_t)((lo >> 58) & 3), nreg = (uint32_t)((lo >> 60) & 15);
        if (!nreg) nreg = 16;
        size_t qw = flg == 0 ? (size_t)nloop * nreg : flg == 1 ? ((size_t)nloop * nreg + 1) / 2 : nloop;
        pos = std::min(g_p3q.size(), pos + 16 + qw * 16);
        if ((lo >> 15) & 1) break;
    }
    return std::min(pos, g_p3q.size());
}

static void p3_release(bool one) {
    if (g_p3pos >= g_p3q.size()) return;
    size_t end = one ? p3_packet_end(g_p3pos) : g_p3q.size();
    gif_feed(GIF_PATH3, &g_p3q[g_p3pos], (uint32_t)((end - g_p3pos) / 16));
    g_p3pos = end;
    if (g_p3pos >= g_p3q.size()) { g_p3q.clear(); g_p3pos = 0; }
}

static void path3_mask(bool m) {
    g_p3_masked = m;
    if (!m) p3_release(true);
}

static void path3_push(const uint8_t *p, uint32_t qwc) {
    // Always queue: the game kicks the GIF chain right before the VIF1 chain, whose first
    // commands mask PATH3 again; anything still unmasked goes out at the next VIF1 command
    // or vblank (path3_idle).
    g_p3q.insert(g_p3q.end(), p, p + (size_t)qwc * 16);
    if (g_p3q.size() - g_p3pos > (64u << 20)) p3_release(false);   // never let a stuck mask grow it unbounded
}

void path3_idle() { if (!g_p3_masked) p3_release(false); }

// ================================================================ VIF
struct Vif {
    int index;
    std::vector<uint32_t> in;      // words not yet consumed
    uint32_t cl = 1, wl = 1, mode = 0, mask = 0, row[4] = {}, col[4] = {};
    uint32_t base = 0, ofst = 0, tops = 0, itops = 0, dbf = 0, mark = 0;
    uint64_t irq_count = 0, unpack_count = 0, mscal_count = 0;
};
static Vif g_vif[2] = {{0}, {1}};

static VuState &vif_vu(Vif &v) { return v.index ? g_vu1 : g_vu0; }

static void vif_start_micro(Vif &v, uint32_t pc_bytes) {
    VuState &vu = vif_vu(v);
    if (v.index == 1) {
        vu.top = v.tops; vu.itop = v.itops;
        if (v.dbf) { v.dbf = 0; v.tops = v.base; } else { v.dbf = 1; v.tops = v.base + v.ofst; }
    }
    v.mscal_count++;
    if (v.index == 0 && !vu.regs) return;
    vu_execute(vu, pc_bytes);
}

// number of 32-bit words an UNPACK consumes
static uint32_t unpack_words(const Vif &v, uint32_t cmd, uint32_t num) {
    uint32_t vn = (cmd >> 2) & 3, vl = cmd & 3;
    uint32_t bits_per = (vl == 3) ? 16 : ((32u >> vl) * (vn + 1));
    uint32_t inputs = num;
    if (v.wl > v.cl && v.wl) inputs = (num / v.wl) * v.cl + std::min(num % v.wl, v.cl);
    return (inputs * bits_per + 31) / 32;
}

static void vif_unpack(Vif &v, uint32_t code, const uint32_t *data) {
    uint32_t cmd = (code >> 24) & 0x7F, num = (code >> 16) & 0xFF, imm = code & 0xFFFF;
    if (!num) num = 256;
    uint32_t vn = (cmd >> 2) & 3, vl = cmd & 3;
    bool masked = (cmd & 0x10) != 0, usn = (imm & 0x4000) != 0, flg = (imm & 0x8000) != 0;
    uint32_t addr = imm & 0x3FF;
    if (v.index == 1 && flg) addr += v.tops;
    uint8_t *mem = v.index ? g_vu1_mem : g_vu0_mem;
    uint32_t qmask = v.index ? 0x3FF : 0xFF;
    const uint8_t *src = (const uint8_t *)data;
    uint32_t bitpos = 0;
    auto rd = [&](int nbits) -> uint32_t {
        uint32_t byte = bitpos / 8, val = 0;
        memcpy(&val, src + byte, nbits / 8);
        bitpos += nbits;
        if (nbits == 32) return val;
        if (usn) return val & ((1u << nbits) - 1);
        return (uint32_t)((int32_t)(val << (32 - nbits)) >> (32 - nbits));
    };
    uint32_t cl = v.cl ? v.cl : 1, wl = v.wl ? v.wl : 1;
    v.unpack_count++;
    if (!masked && v.mode == 0 && wl <= cl) {
        // common case: every element has input, no mask, no row arithmetic
        const uint32_t nc = vl == 3 ? 1 : vn + 1;
        for (uint32_t n = 0; n < num; n++) {
            uint32_t qw = wl == cl ? addr + n : addr + (n / wl) * cl + n % wl;
            uint8_t *dst = mem + (qw & qmask) * 16;
            uint32_t c[4];
            uint32_t ncomp = nc;
            switch (vl) {
            case 0: memcpy(c, src + (size_t)n * nc * 4, nc * 4); break;
            case 1:
                for (uint32_t k = 0; k < nc; k++) {
                    uint16_t h; memcpy(&h, src + ((size_t)n * nc + k) * 2, 2);
                    c[k] = usn ? (uint32_t)h : (uint32_t)(int32_t)(int16_t)h;
                }
                break;
            case 2:
                for (uint32_t k = 0; k < nc; k++) {
                    uint8_t b = src[(size_t)n * nc + k];
                    c[k] = usn ? (uint32_t)b : (uint32_t)(int32_t)(int8_t)b;
                }
                break;
            default: {
                uint16_t p; memcpy(&p, src + (size_t)n * 2, 2);
                c[0] = (p & 31) << 3; c[1] = ((p >> 5) & 31) << 3; c[2] = ((p >> 10) & 31) << 3; c[3] = (uint32_t)(p >> 15) << 7;
                ncomp = 4;
                break;
            }
            }
            if (vl != 3 && vn == 0) { c[1] = c[2] = c[3] = c[0]; ncomp = 4; }
            memcpy(dst, c, ncomp * 4);
        }
        return;
    }
    for (uint32_t n = 0; n < num; n++) {
        uint32_t cycle = n % wl;
        uint32_t qw = (wl <= cl) ? addr + (n / wl) * cl + cycle : addr + n;
        bool has_input = wl <= cl || cycle < cl;
        uint32_t comp[4] = {0, 0, 0, 0};
        int ncomp = 0;
        if (has_input) {
            if (vl == 3) {   // V4-5: RGBA 5551
                uint32_t p = rd(16) & 0xFFFF;
                comp[0] = (p & 31) << 3; comp[1] = ((p >> 5) & 31) << 3; comp[2] = ((p >> 10) & 31) << 3; comp[3] = (p >> 15) << 7;
                ncomp = 4;
            } else {
                int nb = 32 >> vl;
                for (uint32_t k = 0; k <= vn; k++) comp[k] = rd(nb);
                ncomp = (int)vn + 1;
                if (vn == 0) { comp[1] = comp[2] = comp[3] = comp[0]; ncomp = 4; }
            }
        }
        uint8_t *dst = mem + ((qw & qmask) * 16);
        uint32_t mrow = std::min(cycle, 3u);
        for (int k = 0; k < 4; k++) {
            uint32_t m = masked ? (v.mask >> ((mrow * 4 + k) * 2)) & 3 : 0;
            uint32_t val;
            if (m == 0) {
                if (!has_input || k >= ncomp) continue;
                val = comp[k];
                if (v.mode == 1) val += v.row[k];
                else if (v.mode == 2) { v.row[k] += val; val = v.row[k]; }
            } else if (m == 1) val = v.row[k];
            else if (m == 2) val = v.col[mrow];
            else continue;
            memcpy(dst + k * 4, &val, 4);
        }
    }
}

// Process as many complete VIF commands as are buffered.
static void vif_run(Vif &v) {
    size_t pos = 0;
    std::vector<uint32_t> &in = v.in;
    while (pos < in.size()) {
        uint32_t code = in[pos];
        uint32_t cmd = (code >> 24) & 0x7F, num = (code >> 16) & 0xFF, imm = code & 0xFFFF;
        size_t need = 1;
        if (cmd == 0x20) need = 2;
        else if (cmd == 0x30 || cmd == 0x31) need = 5;
        else if (cmd == 0x4A) need = 1 + (size_t)(num ? num : 256) * 2;
        else if (cmd == 0x50 || cmd == 0x51) need = 1 + (size_t)(imm ? imm : 65536) * 4;
        else if (cmd >= 0x60) need = 1 + unpack_words(v, cmd, num ? num : 256);
        // DIRECT and UNPACK data are qword-aligned in the stream relative to the command
        if (pos + need > in.size()) break;
        const uint32_t *d = &in[pos + 1];
        // while PATH3 is left unmasked it competes freely; flush it before anything that
        // draws (DIRECT, micro program) or waits on it (FLUSHA)
        if (!g_p3_masked && (cmd == 0x13 || cmd == 0x14 || cmd == 0x15 || cmd == 0x17 || cmd == 0x50 || cmd == 0x51)) p3_release(false);
        switch (cmd) {
        case 0x00: break;                                              // NOP
        case 0x01: v.cl = imm & 0xFF; v.wl = (imm >> 8) & 0xFF; break;  // STCYCL
        case 0x02: v.ofst = imm & 0x3FF; v.dbf = 0; v.tops = v.base; break;   // OFFSET
        case 0x03: v.base = imm & 0x3FF; break;                        // BASE
        case 0x04: v.itops = imm & 0x3FF; break;                       // ITOP
        case 0x05: v.mode = imm & 3; break;                            // STMOD
        case 0x06: path3_mask((imm & 0x8000) != 0); break;            // MSKPATH3
        case 0x07: v.mark = imm; break;                                // MARK
        case 0x10: case 0x11: case 0x13: break;                        // FLUSHE / FLUSH / FLUSHA
        case 0x14: case 0x15: vif_start_micro(v, imm * 8); break;      // MSCAL / MSCALF
        case 0x17: vif_start_micro(v, vif_vu(v).tpc); break;           // MSCNT
        case 0x20: v.mask = d[0]; break;                               // STMASK
        case 0x30: memcpy(v.row, d, 16); break;                        // STROW
        case 0x31: memcpy(v.col, d, 16); break;                        // STCOL
        case 0x4A: {                                                   // MPG
            uint8_t *micro = v.index ? g_vu1_micro : g_vu0_micro;
            uint32_t mask = v.index ? 0x3FFF : 0xFFF;
            uint32_t n = num ? num : 256;
            bool changed = false;
            for (uint32_t k = 0; k < n * 2; k++) {
                uint8_t *dst = micro + ((imm * 8 + k * 4) & mask);
                if (memcmp(dst, &d[k], 4)) { memcpy(dst, &d[k], 4); changed = true; }
            }
            if (changed) vu_micro_dirty(v.index);
            break;
        }
        case 0x50: case 0x51:                                          // DIRECT / DIRECTHL
            if (v.index == 1) gif_feed(GIF_PATH2, (const uint8_t *)d, imm ? imm : 65536);
            break;
        default:
            if (cmd >= 0x60) vif_unpack(v, code, d);
            else if (rt_once(0xC10000000ull | (v.index << 8) | cmd)) rt_log(LOG_WARN, "VIF%d: unknown command %02X (%08X)", v.index, cmd, code);
        }
        pos += need;
        if (code & 0x80000000u) {   // I bit: interrupt after this command
            v.irq_count++;
            kernel_intc_event(v.index ? INTC_VIF1 : INTC_VIF0);
        }
    }
    in.erase(in.begin(), in.begin() + pos);
}

static void vif_push(Vif &v, const uint8_t *p, uint32_t bytes) {
    size_t o = v.in.size();
    v.in.resize(o + bytes / 4);
    memcpy(&v.in[o], p, bytes & ~3u);
}

// ================================================================ DMA
static uint64_t g_dma_count[10];
#include <chrono>
uint64_t g_prof_vu_ns, g_prof_draw_ns;
static uint64_t g_prof_dma_ns;

static void dma_sink(int ch, const uint8_t *p, uint32_t qwc) {
    if (!qwc) return;
    switch (ch) {
    case 0: case 1: vif_push(g_vif[ch], p, qwc * 16); break;
    case 2: path3_push(p, qwc); break;
    default: break;
    }
}

static void dma_transfer_impl(int ch, uint32_t chcr, uint32_t madr, uint32_t qwc, uint32_t tadr);
void dma_transfer(int ch, uint32_t chcr, uint32_t madr, uint32_t qwc, uint32_t tadr) {
    auto t0 = std::chrono::steady_clock::now();
    dma_transfer_impl(ch, chcr, madr, qwc, tadr);
    g_prof_dma_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}
static void dma_transfer_impl(int ch, uint32_t chcr, uint32_t madr, uint32_t qwc, uint32_t tadr) {
    g_dma_count[ch]++;
    if (g_dma_count[ch] == 1)
        rt_log(LOG_INFO, "first DMA on channel %d (chcr=%08X madr=%08X qwc=%X tadr=%08X)", ch, chcr, madr, qwc, tadr);
    static const uint32_t base[] = {0x10008000, 0x10009000, 0x1000A000, 0x1000B000, 0x1000B400,
                                    0x1000C000, 0x1000C400, 0x1000C800, 0x1000D000, 0x1000D400};
    uint32_t mod = (chcr >> 2) & 3;
    bool tte = (chcr >> 6) & 1, tie = (chcr >> 7) & 1;
    if (ch == 8 || ch == 9) {   // fromSPR / toSPR: normal mode copies with SADR
        uint32_t sadr = hw_reg(base[ch] + 0x80) & 0x3FF0;
        for (uint32_t k = 0; k < qwc; k++) {
            uint8_t *ram = dma_ptr(madr + k * 16, false), *spr = g_spr + ((sadr + k * 16) & 0x3FF0);
            if (ch == 8) memcpy(ram, spr, 16); else memcpy(spr, ram, 16);
        }
        hw_set_reg(base[ch] + 0x10, madr + qwc * 16);
        return;
    }
    if (ch > 2) return;   // IPU / SIF channels are handled elsewhere or not yet emulated
    if (mod == 0) {
        dma_sink(ch, dma_ptr(madr, false), qwc);
        if (ch <= 1) vif_run(g_vif[ch]);
        hw_set_reg(base[ch] + 0x10, madr + qwc * 16);
        return;
    }
    // source chain
    if (qwc) { dma_sink(ch, dma_ptr(madr, false), qwc); }
    uint32_t asr[2] = {0, 0};
    int asp = 0;
    uint64_t last_tag = 0;
    for (int guard = 0; guard < 1000000; guard++) {
        uint64_t tag;
        uint8_t *tp = dma_ptr(tadr, false);
        memcpy(&tag, tp, 8);
        last_tag = tag;
        uint32_t tqwc = (uint32_t)(tag & 0xFFFF), id = (uint32_t)((tag >> 28) & 7), irq = (uint32_t)((tag >> 31) & 1);
        uint32_t addr = (uint32_t)((tag >> 32) & 0x7FFFFFF0);
        bool spr = (tag >> 63) & 1;
        if (tte && ch <= 1) vif_push(g_vif[ch], tp + 8, 8);
        bool end = false;
        uint32_t next = tadr + 16;
        switch (id) {
        case 0: dma_sink(ch, dma_ptr(addr, spr), tqwc); end = true; break;                         // refe
        case 1: dma_sink(ch, dma_ptr(tadr + 16, false), tqwc); next = tadr + 16 + tqwc * 16; break;  // cnt
        case 2: dma_sink(ch, dma_ptr(tadr + 16, false), tqwc); next = addr; break;                 // next
        case 3: case 4: dma_sink(ch, dma_ptr(addr, spr), tqwc); next = tadr + 16; break;           // ref / refs
        case 5:                                                                                    // call
            dma_sink(ch, dma_ptr(tadr + 16, false), tqwc);
            if (asp < 2) asr[asp++] = tadr + 16 + tqwc * 16;
            next = addr; break;
        case 6:                                                                                    // ret
            dma_sink(ch, dma_ptr(tadr + 16, false), tqwc);
            if (asp > 0) next = asr[--asp]; else end = true;
            break;
        case 7: dma_sink(ch, dma_ptr(tadr + 16, false), tqwc); end = true; break;                  // end
        }
        if (ch <= 1) vif_run(g_vif[ch]);
        tadr = next;
        if (end || (irq && tie)) break;
    }
    hw_set_reg(base[ch] + 0x30, tadr);
    hw_set_reg(base[ch], (hw_reg(base[ch]) & 0xFFFF) | (uint32_t)((last_tag >> 16) & 0xFFFF0000u));
}

void gif_fifo_write32(uint32_t) {}

extern uint64_t g_prof_pixels, g_prof_prims[8], g_vu_instr;
std::string dma_debug_state() {
    char b[512];
    snprintf(b, sizeof b, "px %llu prims tri %llu strip %llu fan %llu sprite %llu line %llu | time ms: dma %llu vu %llu draw %llu | dma vif1=%llu gif=%llu | vif1 irq=%llu unpack=%llu mscal=%llu vuinstr=%llu",
             (unsigned long long)g_prof_pixels, (unsigned long long)g_prof_prims[3], (unsigned long long)g_prof_prims[4],
             (unsigned long long)g_prof_prims[5], (unsigned long long)g_prof_prims[6], (unsigned long long)(g_prof_prims[1] + g_prof_prims[2]),
             (unsigned long long)(g_prof_dma_ns / 1000000), (unsigned long long)(g_prof_vu_ns / 1000000), (unsigned long long)(g_prof_draw_ns / 1000000),
             (unsigned long long)g_dma_count[1], (unsigned long long)g_dma_count[2],
             (unsigned long long)g_vif[1].irq_count, (unsigned long long)g_vif[1].unpack_count,
             (unsigned long long)g_vif[1].mscal_count, (unsigned long long)g_vu_instr);
    return b;
}
