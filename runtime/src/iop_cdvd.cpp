// CD/DVD drive (cdvdman/cdvdfsv) at the SIF RPC level, backed by the user's ISO.
#include "rt_internal.h"
#include "sif.h"
#include "iso.h"
#include <algorithm>
#include <vector>

enum : uint32_t {
    CD_INIT = 0x80000592, CD_SCMD = 0x80000593, CD_NCMD = 0x80000595,
    CD_SEARCHFILE = 0x80000597, CD_DISKREADY = 0x8000059A, CD_UNK59C = 0x8000059C,
};
enum { SCECdPS2DVD = 0x14, SCECdComplete = 0x02 };

static uint32_t rd32(SifRpcCall &c, uint32_t off) {
    uint32_t v; memcpy(&v, g_iop_ram + ((c.buff + off) & 0x1FFFFF), 4); return v;
}
static void reply(SifRpcCall &c, const uint32_t *w, uint32_t nwords) {
    if (c.recv_addr && c.recv_size) sif_write_ee(c.recv_addr, w, std::min<uint32_t>(c.recv_size, nwords * 4));
}
static void reply1(SifRpcCall &c, uint32_t v0, uint32_t v1 = 0) { uint32_t w[4] = {v0, v1, 0, 0}; reply(c, w, 4); }

static uint8_t bcd(int v) { return (uint8_t)(((v / 10) << 4) | (v % 10)); }

static void srv_init(SifRpcCall &c) { reply1(c, 1); }

static void srv_scmd(SifRpcCall &c) {
    switch (c.fno) {
    case 0x01: {  // ReadClock: {result, stat, second, minute, hour, pad, day, month, year} (BCD)
        uint8_t clk[12] = {1, 0, 0, 0, 0, bcd(0), bcd(0), bcd(12), 0, bcd(1), bcd(1), bcd(8)};
        if (c.recv_addr) sif_write_ee(c.recv_addr, clk, std::min<uint32_t>(c.recv_size, sizeof clk));
        return;
    }
    case 0x03: reply1(c, SCECdPS2DVD); return;   // GetDiskType
    case 0x04: reply1(c, 0); return;             // GetError: none
    case 0x05: reply1(c, 1, 0); return;          // TrayReq: ok, tray not changed
    case 0x0C: reply1(c, 0x0A); return;          // Status: spinning, paused/ready
    case 0x22: reply1(c, 1); return;             // MMode
    default:
        if (rt_once(0x1400000000ull | c.fno)) rt_log(LOG_INFO, "cdvd: S-command %X", c.fno);
        reply1(c, 1);
    }
}

// Movies (.PSS) need the IPU, which isn't emulated yet: report them missing so the game
// skips them. -movies turns the lookup back on.
bool g_movies;
static bool is_movie(const char *n) {
    size_t l = strlen(n);
    while (l && (n[l - 1] == '1' || n[l - 1] == ';')) l--;
    return l >= 4 && _strnicmp(n + l - 4, ".PSS", 4) == 0;
}

// SearchFile request: sceCdlFILE (32 bytes) | u32 | char name[256] | u32 dest (EE) | u32
static void srv_searchfile(SifRpcCall &c) {
    const char *name = (const char *)(g_iop_ram + ((c.buff + 0x24) & 0x1FFFFF));
    uint32_t dest = rd32(c, 0x124);
    IsoEntry e;
    bool ok = (g_movies || !is_movie(name)) && iso_lookup(name, e) && !e.is_dir;
    rt_log(ok ? LOG_INFO : LOG_WARN, "sceCdSearchFile(%s) -> %s lsn %u size %u (dest %08X)", name,
           ok ? "found" : "NOT FOUND", e.lsn, e.size, dest);
    if (ok && dest) {
        uint8_t f[32] = {};
        memcpy(f + 0, &e.lsn, 4);
        memcpy(f + 4, &e.size, 4);
        std::string n = e.name + ";1";
        memcpy(f + 8, n.c_str(), std::min<size_t>(n.size(), 15));
        sif_write_ee(dest, f, 32);
    }
    reply1(c, ok ? 1 : 0);
}

// N-commands: reads etc. Data goes straight into EE memory; completion is
// signalled by the RPC reply (libcdvd's sceCdSync waits on it).
static void srv_ncmd(SifRpcCall &c) {
    switch (c.fno) {
    case 0x01: case 0x03: {  // Read / DvdRead: {lsn, sectors, ee_buf, mode...}
        uint32_t lsn = rd32(c, 0), n = rd32(c, 4), buf = rd32(c, 8);
        std::vector<uint8_t> data((size_t)n * 2048);
        bool ok = iso_read_sectors(lsn, n, data.data());
        rt_log(LOG_TRACE, "sceCdRead(lsn %u, %u sectors) -> EE %08X", lsn, n, buf);
        if (rt_once(0x1500000000ull)) rt_log(LOG_INFO, "first sceCdRead(lsn %u, %u sectors) -> EE %08X", lsn, n, buf);
        if (ok) sif_write_ee(buf, data.data(), (uint32_t)data.size());
        reply1(c, ok ? 1 : 0);
        return;
    }
    default:
        if (rt_once(0x1600000000ull | c.fno)) {
            uint32_t w[4]; for (int k = 0; k < 4; k++) w[k] = rd32(c, k * 4);
            rt_log(LOG_INFO, "cdvd: N-command %X args %08X %08X %08X %08X", c.fno, w[0], w[1], w[2], w[3]);
        }
        reply1(c, 1);
    }
}

static void srv_diskready(SifRpcCall &c) { reply1(c, SCECdComplete); }

void iop_cdvd_init() {
    sif_register_server(CD_INIT, "cdvd-init", srv_init);
    sif_register_server(CD_SCMD, "cdvd-scmd", srv_scmd);
    sif_register_server(CD_NCMD, "cdvd-ncmd", srv_ncmd);
    sif_register_server(CD_SEARCHFILE, "cdvd-search", srv_searchfile);
    sif_register_server(CD_DISKREADY, "cdvd-diskready", srv_diskready);
    sif_register_server(CD_UNK59C, "cdvd-59c", srv_diskready);
}
