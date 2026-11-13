// Host game controllers (XInput: Xbox pads, and anything Steam Input / DS4Windows exposes as one)
// mapped onto the DualShock 2 the game expects. XInput is loaded at runtime, so the game still
// runs on systems without it (keyboard only).
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <xinput.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include "rt_internal.h"

typedef DWORD(WINAPI *GetStateFn)(DWORD, XINPUT_STATE *);
typedef DWORD(WINAPI *SetStateFn)(DWORD, XINPUT_VIBRATION *);
static GetStateFn g_get;
static SetStateFn g_set;
static bool g_tried;

struct HostPad { bool connected = false; uint32_t retry = 0; uint16_t rumble_lo = 0, rumble_hi = 0; bool rumble_dirty = false; };
static HostPad g_host[4];

static void load_xinput() {
    g_tried = true;
    for (const char *dll : {"xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"}) {
        if (HMODULE m = LoadLibraryA(dll)) {
            g_get = (GetStateFn)GetProcAddress(m, "XInputGetState");
            g_set = (SetStateFn)GetProcAddress(m, "XInputSetState");
            if (g_get) { rt_log(LOG_INFO, "gamepad: using %s", dll); return; }
        }
    }
    rt_log(LOG_INFO, "gamepad: XInput not available, keyboard only");
}

// Stick -> DualShock axes. The game applies its own DualShock-sized deadzone, so only a small
// radial one is applied here (enough to hide stick drift). Stacking XInput's recommended
// 24-27% deadzones on top of the game's made the sticks feel unresponsive and then jumpy.
// NPPL_DEADZONE=<0-30> sets it in percent of full throw (default 8).
static void stick_axes(SHORT sx, SHORT sy, uint8_t &ox, uint8_t &oy) {
    static const float dz = [] {
        const char *e = getenv("NPPL_DEADZONE");
        return std::clamp(e ? (float)atof(e) : 8.0f, 0.0f, 30.0f) / 100.0f;
    }();
    float x = std::max(sx / 32767.0f, -1.0f), y = std::max(sy / 32767.0f, -1.0f);
    float m = sqrtf(x * x + y * y);
    if (m <= dz) { x = y = 0; }
    else {
        float k = std::min((m - dz) / (1.0f - dz), 1.0f) / m;   // rescale so full throw stays full
        x *= k; y *= k;
    }
    auto conv = [](float v) { return (uint8_t)std::clamp((int)lroundf(v * 127.5f + 127.5f), 0, 255); };
    ox = conv(x);
    oy = conv(-y);   // PS2: 0 = up
}

// Poll host controller `idx` (0-3). Returns false if it isn't connected. buttons uses the
// DualShock bit layout (active high); axes = right x, right y, left x, left y (0x80 = centre).
bool gamepad_poll(int idx, uint32_t &buttons, uint8_t axes[4]) {
    if (!g_tried) load_xinput();
    if (!g_get || idx < 0 || idx > 3) return false;
    HostPad &h = g_host[idx];
    // probing an empty slot is slow: retry disconnected slots about once a second
    if (!h.connected && h.retry && --h.retry) return false;
    XINPUT_STATE st;
    if (g_get((DWORD)idx, &st) != ERROR_SUCCESS) {
        if (h.connected) rt_log(LOG_INFO, "gamepad: controller %d disconnected", idx);
        h.connected = false; h.retry = 60;
        return false;
    }
    if (!h.connected) { rt_log(LOG_INFO, "gamepad: controller %d connected", idx); h.connected = true; h.rumble_dirty = true; }
    const XINPUT_GAMEPAD &g = st.Gamepad;
    WORD w = g.wButtons;
    uint32_t b = 0;
    if (w & XINPUT_GAMEPAD_BACK) b |= 1u << 0;            // SELECT
    if (w & XINPUT_GAMEPAD_LEFT_THUMB) b |= 1u << 1;      // L3
    if (w & XINPUT_GAMEPAD_RIGHT_THUMB) b |= 1u << 2;     // R3
    if (w & XINPUT_GAMEPAD_START) b |= 1u << 3;           // START
    if (w & XINPUT_GAMEPAD_DPAD_UP) b |= 1u << 4;
    if (w & XINPUT_GAMEPAD_DPAD_RIGHT) b |= 1u << 5;
    if (w & XINPUT_GAMEPAD_DPAD_DOWN) b |= 1u << 6;
    if (w & XINPUT_GAMEPAD_DPAD_LEFT) b |= 1u << 7;
    if (g.bLeftTrigger > 40) b |= 1u << 8;                // L2
    if (g.bRightTrigger > 40) b |= 1u << 9;               // R2
    if (w & XINPUT_GAMEPAD_LEFT_SHOULDER) b |= 1u << 10;  // L1
    if (w & XINPUT_GAMEPAD_RIGHT_SHOULDER) b |= 1u << 11; // R1
    if (w & XINPUT_GAMEPAD_Y) b |= 1u << 12;              // triangle
    if (w & XINPUT_GAMEPAD_B) b |= 1u << 13;              // circle
    if (w & XINPUT_GAMEPAD_A) b |= 1u << 14;              // cross
    if (w & XINPUT_GAMEPAD_X) b |= 1u << 15;              // square
    buttons = b;
    stick_axes(g.sThumbRX, g.sThumbRY, axes[0], axes[1]);   // right x, y
    stick_axes(g.sThumbLX, g.sThumbLY, axes[2], axes[3]);   // left x, y
    if (h.rumble_dirty && g_set) {
        XINPUT_VIBRATION v{h.rumble_lo, h.rumble_hi};
        g_set((DWORD)idx, &v);
        h.rumble_dirty = false;
    }
    return true;
}

bool gamepad_connected(int idx) { return idx >= 0 && idx < 4 && g_host[idx].connected; }

// DualShock motors: small = on/off (high-frequency motor), big = 0-255 (low-frequency motor).
void gamepad_rumble(int idx, bool small_on, uint8_t big) {
    if (idx < 0 || idx > 3) return;
    HostPad &h = g_host[idx];
    uint16_t lo = (uint16_t)(big * 257), hi = small_on ? 0xFFFF : 0;
    if (lo != h.rumble_lo || hi != h.rumble_hi) { h.rumble_lo = lo; h.rumble_hi = hi; h.rumble_dirty = true; }
}
