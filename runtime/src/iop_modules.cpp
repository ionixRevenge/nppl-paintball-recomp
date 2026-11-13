// High-level IOP modules. Each SCE server the game binds to is answered here.
// First version: log every call and reply with zeroed data ("success"), so
// the game's startup can proceed and the log shows what each module must do.
#include "rt_internal.h"
#include "sif.h"
#include <algorithm>
#include <vector>

void pad_open(uint32_t port, uint32_t slot, uint32_t ee_area);  // pad.cpp
bool gamepad_connected(int idx);                                 // gamepad.cpp
void gamepad_rumble(int idx, bool small_on, uint8_t big);

static void reply_zero(SifRpcCall &c) {
    if (!c.recv_addr || !c.recv_size) return;
    std::vector<uint8_t> z(c.recv_size, 0);
    sif_write_ee(c.recv_addr, z.data(), c.recv_size);
}

static void hexdump_call(SifRpcCall &c, const char *mod) {
    const uint8_t *in = g_iop_ram + (c.buff & 0x1FFFFF);
    uint32_t n = std::min<uint32_t>(c.send_size, 0x130);
    char line[2048]; int o = 0;
    for (uint32_t k = 0; k < n; k++) {
        uint8_t b = in[k];
        o += snprintf(line + o, sizeof line - o, "%02X", b);
        if (k % 4 == 3) line[o++] = ' ';
    }
    line[o] = 0;
    char txt[0x140]; uint32_t t = 0;
    for (uint32_t k = 0; k < n && t < 0x130; k++) txt[t++] = (in[k] >= 32 && in[k] < 127) ? (char)in[k] : '.';
    txt[t] = 0;
    rt_log(LOG_INFO, "IOP %s %08X fn %u send %X recv %X: %s |%s|", mod, c.sid, c.fno, c.send_size, c.recv_size, line, txt);
}

static void log_call(SifRpcCall &c, const char *mod) {
    if (!strcmp(mod, "fileio") || !strcmp(mod, "unknown") || !strcmp(mod, "cdvd")) {
        static int n; if (n++ < 60) hexdump_call(c, mod);
        return;
    }
    if (rt_once(0xF00000000ull | ((uint64_t)(c.sid & 0xFFFF) << 12) | (c.fno & 0xFFF))) {
        const uint8_t *in = g_iop_ram + (c.buff & 0x1FFFFF);
        uint32_t w[4] = {};
        memcpy(w, in, 16);
        rt_log(LOG_INFO, "IOP %s: fn %u (send %X, recv %X) data %08X %08X %08X %08X", mod, c.fno, c.send_size,
               c.recv_size, w[0], w[1], w[2], w[3]);
    }
}

#define GENERIC(name, label) static void name(SifRpcCall &c) { log_call(c, label); reply_zero(c); }
GENERIC(srv_fileio, "fileio")
GENERIC(srv_heap, "iopheap")
// loadfile: modules are "loaded" (their functionality is provided natively).
// Reply overlays the request: word 0 = module id (> 0 on success), word 1 = result.
static void srv_loadfile(SifRpcCall &c) {
    static int next_id = 1;
    const char *path = (const char *)(g_iop_ram + (c.buff & 0x1FFFFF) + 8);
    uint32_t reply[2] = {0, 0};
    if (c.fno == 255) {
        memcpy(&reply[0], "2710", 4);  // loadfile version; libkernel 2.7.1 compares it with "2710"
    } else if (c.fno <= 6) {
        reply[0] = ++next_id;
        rt_log(LOG_INFO, "loadfile: fn %u load \"%.80s\" -> id %u", c.fno, path, reply[0]);
    } else log_call(c, "loadfile");
    if (c.recv_addr) sif_write_ee(c.recv_addr, reply, std::min<uint32_t>(c.recv_size, 8));
}
GENERIC(srv_pad, "pad")
GENERIC(srv_sdr, "sdr")
GENERIC(srv_unknown, "unknown")


// ---------------------------------------------------------------- padman
// Commands arrive in a 128-byte buffer: word 0 = command, result goes to word 3.
enum { PAD_OPEN = 0x01, PAD_INFO_ACT = 0x02, PAD_INFO_COMB = 0x03, PAD_INFO_MODE = 0x04,
       PAD_SET_MMODE = 0x05, PAD_SET_ACTDIR = 0x06, PAD_SET_ACTALIGN = 0x07, PAD_GET_BTNMASK = 0x08,
       PAD_SET_BTNMASK = 0x09, PAD_SET_VREF = 0x0A, PAD_GET_PORTMAX = 0x0B, PAD_GET_SLOTMAX = 0x0C,
       PAD_CLOSE = 0x0D, PAD_END = 0x0E, PAD_INIT = 0x10, PAD_GET_CONNECT = 0x11, PAD_GET_MODVER = 0x12 };

static void srv_padman(SifRpcCall &c) {
    uint8_t *in = g_iop_ram + (c.buff & 0x1FFFFF);
    uint32_t w[32];
    memcpy(w, in, sizeof w);
    uint32_t cmd = w[0], result = 0;
    switch (cmd) {
    case PAD_GET_MODVER: result = 0x0400; break;   // padman 4.0, matching libpad
    case PAD_INIT: result = 1; break;
    case PAD_GET_PORTMAX: result = 2; break;
    case PAD_GET_SLOTMAX: result = 1; break;
    case PAD_GET_CONNECT: result = (w[1] == 0 || gamepad_connected((int)w[1])) ? 1 : 0; break;   // port 2 needs a second host controller
    case PAD_SET_ACTDIR:   // motor values: data bytes follow port/slot (actuator 0 = small on/off, 1 = big 0-255)
        if (w[1] < 2) gamepad_rumble((int)w[1], in[12] != 0, in[13]);
        result = 1; break;
    case PAD_OPEN:
        rt_log(LOG_INFO, "pad: open port %u slot %u area %08X", w[1], w[2], w[4]);
        pad_open(w[1], w[2], w[4]);
        result = 1; break;
    case PAD_CLOSE: case PAD_END: result = 1; break;
    case PAD_GET_BTNMASK: result = 0xFFFFF; break;
    default:
        if (rt_once(0xF10000000ull | cmd)) rt_log(LOG_INFO, "pad: command %X (args %08X %08X %08X)", cmd, w[1], w[2], w[3]);
        result = 1;
    }
    w[3] = result;
    if (c.recv_addr) sif_write_ee(c.recv_addr, w, std::min<uint32_t>(c.recv_size, sizeof w));
}

void iop_cdvd_init();  // iop_cdvd.cpp
void iop_mc_init();    // iop_mc.cpp
void iop_audio_init(); // iop_audio.cpp

void iop_modules_init() {
    sif_register_server(0x80000100, "padman", srv_padman);
    sif_register_server(0x80000101, "padman-ext", srv_padman);
    sif_register_server(0x80000001, "fileio", srv_fileio);
    sif_register_server(0x80000003, "iopheap", srv_heap);
    sif_register_server(0x80000006, "loadfile", srv_loadfile);
    iop_cdvd_init();   // iop_cdvd.cpp: cdvdman/cdvdfsv servers backed by the ISO
    sif_register_server(0x8000010F, "pad", srv_pad);
    sif_register_server(0x8000011F, "pad-ext", srv_pad);
    iop_mc_init();     // iop_mc.cpp: memory card backed by memcard/port0
    sif_register_server(0x80000701, "sdr", srv_sdr);
    iop_audio_init();  // iop_audio.cpp: AUDIO.IRX replacement + mixer
}

// Servers we don't know yet are created on first bind so their calls get logged.
void iop_register_unknown(uint32_t sid) { sif_register_server(sid, "unknown", srv_unknown); }
