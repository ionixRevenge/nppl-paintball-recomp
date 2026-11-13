// Controller state written into the EE-side libpad buffer (padman does this on hardware).
//
// libpad 2.7 (new padman protocol) gives padman a 256-byte area per port: two 128-byte
// halves, and scePadRead uses whichever half has the larger frame counter. Layout:
//   0x00 data[32]     0x00 = ok, 0x01 = mode id (0x73 = DualShock analog), 0x02-0x03 = buttons
//                     (active low), 0x04-0x07 = right x/y, left x/y
//   0x50 modeTable[4] 0x58 frame   0x60 length
//   0x64 modeConfig, modeCurId, model, buttonDataReady, nrOfModes, modeCurOffs,
//        nrOfActuators, numActComb, val_c6, mode, lock, actComb
//   0x70 state (6 = stable), 0x71 reqState (0 = complete), 0x72 ok
#include "rt_internal.h"
#include <string>
#include <vector>

uint32_t display_keys();   // display.cpp
bool gamepad_poll(int idx, uint32_t &buttons, uint8_t axes[4]);   // gamepad.cpp
bool gamepad_connected(int idx);

struct PadPort { bool open = false; uint32_t area = 0; uint32_t frame = 0; };
static PadPort g_pad[2];
uint32_t g_pad_script;   // debug: bitmask pressed in a pulse every 2 s (main.cpp -press)

// Debug input script (main.cpp -input): "frame:BUTTONS[:len],..." where frame counts vblanks
// from boot, BUTTONS is names joined by '+' (START, SELECT, X, O, TRI, SQ, UP, DOWN, LEFT,
// RIGHT, L1, R1, L2, R2, L3, R3) or a hex mask, and len defaults to 6 frames.
struct PadEvent { uint32_t frame, len, mask; };
static std::vector<PadEvent> g_pad_events;

static uint32_t button_mask(const std::string &name) {
    static const struct { const char *n; uint32_t m; } k[] = {
        {"SELECT", 0x1}, {"L3", 0x2}, {"R3", 0x4}, {"START", 0x8}, {"UP", 0x10}, {"RIGHT", 0x20},
        {"DOWN", 0x40}, {"LEFT", 0x80}, {"L2", 0x100}, {"R2", 0x200}, {"L1", 0x400}, {"R1", 0x800},
        {"TRI", 0x1000}, {"O", 0x2000}, {"X", 0x4000}, {"SQ", 0x8000},
        // analog sticks pushed fully in one direction
        {"LUP", 1u << 16}, {"LDOWN", 1u << 17}, {"LLEFT", 1u << 18}, {"LRIGHT", 1u << 19},
        {"RUP", 1u << 20}, {"RDOWN", 1u << 21}, {"RLEFT", 1u << 22}, {"RRIGHT", 1u << 23}};
    for (auto &e : k) if (_stricmp(name.c_str(), e.n) == 0) return e.m;
    return (uint32_t)strtoul(name.c_str(), nullptr, 16);
}

void pad_set_script(const char *spec) {
    std::string s = spec;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        std::string ev = s.substr(i, j - i);
        size_t c1 = ev.find(':'), c2 = c1 == std::string::npos ? c1 : ev.find(':', c1 + 1);
        if (c1 != std::string::npos) {
            PadEvent e{(uint32_t)strtoul(ev.c_str(), nullptr, 10), 6, 0};
            std::string btn = ev.substr(c1 + 1, c2 == std::string::npos ? std::string::npos : c2 - c1 - 1);
            if (c2 != std::string::npos) e.len = (uint32_t)strtoul(ev.c_str() + c2 + 1, nullptr, 10);
            for (size_t b = 0; b <= btn.size();) {
                size_t p = btn.find('+', b);
                if (p == std::string::npos) p = btn.size();
                e.mask |= button_mask(btn.substr(b, p - b));
                b = p + 1;
            }
            g_pad_events.push_back(e);
        }
        i = j + 1;
    }
    rt_log(LOG_INFO, "pad: %zu scripted input events", g_pad_events.size());
}

void pad_open(uint32_t port, uint32_t slot, uint32_t ee_area) {
    if (port < 2) { g_pad[port].open = true; g_pad[port].area = ee_area & 0x1FFFFFF; }
    (void)slot;
}

static void fill_half(uint8_t *p, uint32_t frame, bool connected, uint32_t keys, const uint8_t axes[4]) {
    if (!connected) {
        memset(p, 0xFF, 32);
        memcpy(p + 0x58, &frame, 4);
        p[0x70] = 0;   // disconnected
        p[0x71] = 0;
        p[0x72] = 0;
        return;
    }
    memset(p, 0, 32);
    p[0] = 0x00;
    p[1] = 0x73;
    uint16_t b = (uint16_t)~keys;
    p[2] = (uint8_t)b;
    p[3] = (uint8_t)(b >> 8);
    memcpy(p + 4, axes, 4);   // right x, right y, left x, left y
    uint32_t len = 32;
    uint16_t modes[4] = {4, 7, 0, 0};   // extended ids: digital, DualShock
    memcpy(p + 0x50, modes, 8);
    static const uint8_t act[8] = {0, 1, 0, 1, 1, 1, 1, 20};   // small motor, big motor
    memcpy(p + 0x30, act, 8);
    memcpy(p + 0x58, &frame, 4);
    memcpy(p + 0x60, &len, 4);
    p[0x64] = 2;     // modeConfig: mode info available
    p[0x65] = 0x73;  // modeCurId
    p[0x66] = 3;     // model: DualShock 2
    p[0x67] = 1;     // buttonDataReady
    p[0x68] = 2;     // nrOfModes
    p[0x69] = 1;     // modeCurOffs (analog)
    p[0x6A] = 2;     // nrOfActuators
    p[0x6B] = 1;     // numActComb
    p[0x6D] = 1;     // mode: analog
    p[0x6E] = 3;     // lock
    p[0x70] = 6;     // PAD_STATE_STABLE
    p[0x71] = 0;     // PAD_RSTAT_COMPLETE
    p[0x72] = 1;
}

// Called every vblank.
void pad_vsync() {
    static uint32_t n;
    uint32_t keys = display_keys();
    ++n;
    if (g_pad_script && (n % 120) < 6) keys |= g_pad_script;
    for (auto &e : g_pad_events)
        if (n >= e.frame && n < e.frame + e.len) {
            keys |= e.mask;
            if (n == e.frame) rt_log(LOG_INFO, "pad: frame %u press %04X", n, e.mask);
        }
    // port 1: keyboard + first host controller; port 2: second host controller
    uint32_t pk[2] = {keys, 0};
    uint8_t axes[2][4] = {{0x80, 0x80, 0x80, 0x80}, {0x80, 0x80, 0x80, 0x80}};
    bool conn[2] = {true, false};
    for (int port = 0; port < 2; port++) {
        uint32_t b = 0;
        if (gamepad_poll(port, b, axes[port])) { pk[port] |= b; conn[port] = true; }
    }
    // keyboard / script stick bits override the host stick on port 1
    auto stick = [](uint32_t k, int neg, int pos, uint8_t &a) { if (k & (1u << neg)) a = 0x00; else if (k & (1u << pos)) a = 0xFF; };
    stick(keys, 22, 23, axes[0][0]); stick(keys, 20, 21, axes[0][1]);
    stick(keys, 18, 19, axes[0][2]); stick(keys, 16, 17, axes[0][3]);
    pk[0] &= 0xFFFF;
    for (int port = 0; port < 2; port++) {
        PadPort &pp = g_pad[port];
        if (!pp.open || !pp.area) continue;
        pp.frame++;
        uint8_t *base = g_ram + pp.area;
        for (int h = 0; h < 2; h++) fill_half(base + h * 128, pp.frame, conn[port], pk[port], axes[port]);
    }
}
