// Game window: shows the GS display buffer each vblank and turns the keyboard into
// a DualShock (see pad.cpp). Win32 + GDI only, so there are no extra dependencies.
#include "rt_internal.h"
#include "gs.h"
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

static const int MAX_W = 2048, MAX_H = 2048;   // room for GPU screenshots at internal resolution
static std::vector<uint32_t> g_front(MAX_W * MAX_H), g_back(MAX_W * MAX_H);
static int g_fw = 640, g_fh = 448;
static std::mutex g_mx;
static HWND g_hwnd;
static std::atomic<uint32_t> g_keys{0};
static uint64_t g_front_gen;   // bumped for every new frame (under g_mx)

// ---------------------------------------------------------------- output upscaler
// The game renders at PS2 resolution (640x448). Instead of a nearest-neighbour stretch, frames
// are scaled to the window with Catmull-Rom bicubic and a contrast-adaptive sharpen (in the
// spirit of AMD FSR's RCAS), which keeps text and HUD edges crisp. Runs on the UI thread.
// NPPL_FILTER=nearest restores the old blocky stretch; NPPL_SHARPEN=<0..1> (default 0.6).
struct Scaler {
    std::vector<uint32_t> out, tmp;
    int ow = 0, oh = 0;
    uint64_t gen = ~0ull;
};
static Scaler g_scaler;

static inline float cr_w(float t) {   // Catmull-Rom kernel
    t = std::fabs(t);
    if (t < 1) return 1.5f * t * t * t - 2.5f * t * t + 1;
    if (t < 2) return -0.5f * t * t * t + 2.5f * t * t - 4 * t + 2;
    return 0;
}
struct Taps { int i[4]; float w[4]; };
static std::vector<Taps> make_taps(int src, int dst) {
    std::vector<Taps> t(dst);
    float sc = (float)src / (float)dst;
    for (int o = 0; o < dst; o++) {
        float c = (o + 0.5f) * sc - 0.5f;
        int b = (int)std::floor(c);
        float sum = 0;
        for (int k = 0; k < 4; k++) {
            int i = b - 1 + k;
            t[o].i[k] = std::clamp(i, 0, src - 1);
            t[o].w[k] = cr_w(c - (float)i);
            sum += t[o].w[k];
        }
        for (float &w : t[o].w) w /= sum;
    }
    return t;
}
static inline uint32_t pack(float r, float g, float b) {
    auto c = [](float v) { return (uint32_t)std::clamp((int)(v + 0.5f), 0, 255); };
    return (c(r) << 16) | (c(g) << 8) | c(b);
}

// src: w x h (stride MAX_W, 0x00RRGGBB). Result in g_scaler.out (ow x oh).
static void upscale(Scaler &S, const uint32_t *src, int w, int h, int ow, int oh, float sharpen) {
    auto tx = make_taps(w, ow), ty = make_taps(h, oh);
    // horizontal pass: w x h -> ow x h, kept as packed RGB
    S.tmp.resize((size_t)ow * h);
    auto hpass = [&](int y0, int y1) {
        for (int y = y0; y < y1; y++) {
            const uint32_t *row = src + (size_t)y * MAX_W;
            uint32_t *o = &S.tmp[(size_t)y * ow];
            for (int x = 0; x < ow; x++) {
                const Taps &t = tx[x];
                float r = 0, g = 0, b = 0;
                for (int k = 0; k < 4; k++) {
                    uint32_t p = row[t.i[k]];
                    r += t.w[k] * ((p >> 16) & 255); g += t.w[k] * ((p >> 8) & 255); b += t.w[k] * (p & 255);
                }
                o[x] = pack(r, g, b);
            }
        }
    };
    std::vector<uint32_t> &up = S.out;
    up.resize((size_t)ow * oh);
    auto vpass = [&](int y0, int y1) {
        for (int y = y0; y < y1; y++) {
            const Taps &t = ty[y];
            const uint32_t *r0 = &S.tmp[(size_t)t.i[0] * ow], *r1 = &S.tmp[(size_t)t.i[1] * ow],
                           *r2 = &S.tmp[(size_t)t.i[2] * ow], *r3 = &S.tmp[(size_t)t.i[3] * ow];
            uint32_t *o = &up[(size_t)y * ow];
            for (int x = 0; x < ow; x++) {
                float r = 0, g = 0, b = 0;
                const uint32_t p[4] = {r0[x], r1[x], r2[x], r3[x]};
                for (int k = 0; k < 4; k++) {
                    r += t.w[k] * ((p[k] >> 16) & 255); g += t.w[k] * ((p[k] >> 8) & 255); b += t.w[k] * (p[k] & 255);
                }
                o[x] = pack(r, g, b);
            }
        }
    };
    // contrast-adaptive sharpen: push each pixel away from its 4-neighbour mean, limited so it
    // never leaves the local min/max (no halos), weaker where local contrast is already high
    std::vector<uint32_t> sh;
    auto spass = [&](int y0, int y1) {
        for (int y = y0; y < y1; y++) {
            const uint32_t *rc = &up[(size_t)y * ow], *ru = &up[(size_t)std::max(y - 1, 0) * ow], *rd = &up[(size_t)std::min(y + 1, oh - 1) * ow];
            uint32_t *o = &sh[(size_t)y * ow];
            for (int x = 0; x < ow; x++) {
                uint32_t c = rc[x], n[4] = {ru[x], rd[x], rc[std::max(x - 1, 0)], rc[std::min(x + 1, ow - 1)]};
                float res[3];
                for (int ch = 0; ch < 3; ch++) {
                    int s = 16 - ch * 8;
                    float v = (float)((c >> s) & 255), mn = v, mx = v, sum = 0;
                    for (uint32_t q : n) { float u = (float)((q >> s) & 255); mn = std::min(mn, u); mx = std::max(mx, u); sum += u; }
                    float amt = sharpen * (1.0f - (mx - mn) / 255.0f * 0.5f);
                    res[ch] = std::clamp(v + amt * (v - sum * 0.25f), mn, mx);
                }
                o[x] = pack(res[0], res[1], res[2]);
            }
        }
    };
    unsigned nt = std::clamp(std::thread::hardware_concurrency() / 2, 1u, 4u);
    auto par = [&](auto f, int rows) {
        std::vector<std::thread> th;
        for (unsigned k = 0; k < nt; k++) th.emplace_back([&, k] { f(rows * (int)k / (int)nt, rows * (int)(k + 1) / (int)nt); });
        for (auto &t : th) t.join();
    };
    par(hpass, h);
    par(vpass, oh);
    if (sharpen > 0) { sh.resize(up.size()); par(spass, oh); up.swap(sh); }
    S.ow = ow; S.oh = oh;
}
static uint64_t g_frames;

static uint32_t key_bit(WPARAM k);
// Held keys, re-checked against the real keyboard each vblank: a key released while another
// window had focus never sends us WM_KEYUP and would otherwise stay held (endless running).
uint32_t display_keys() {
    uint32_t k = g_keys.load();
    if (!k) return k;
    static const int vks[] = {VK_RETURN, VK_BACK, VK_SHIFT, VK_UP, VK_RIGHT, VK_DOWN, VK_LEFT, 'Q', 'E', '1', '3', 'I', 'L', 'K',
                              VK_SPACE, 'J', 'W', 'S', 'A', 'D', 'T', 'G', 'F', 'H'};
    uint32_t down = 0;
    for (int vk : vks) if (GetAsyncKeyState(vk) & 0x8000) down |= key_bit((WPARAM)vk);
    if (k & ~down) g_keys &= down;
    return k & down;
}

// keyboard -> DualShock button bits (active high here; pad.cpp inverts)
static uint32_t key_bit(WPARAM k) {
    switch (k) {
    case VK_RETURN: return 1u << 3;            // START
    case VK_BACK: case VK_SHIFT: return 1u << 0;   // SELECT
    case VK_UP: return 1u << 4;
    case VK_RIGHT: return 1u << 5;
    case VK_DOWN: return 1u << 6;
    case VK_LEFT: return 1u << 7;
    case 'Q': return 1u << 10;                 // L1
    case 'E': return 1u << 11;                 // R1
    case '1': return 1u << 8;                  // L2
    case '3': return 1u << 9;                  // R2
    case 'I': return 1u << 12;                 // triangle
    case 'L': return 1u << 13;                 // circle
    case 'K': case VK_SPACE: return 1u << 14;  // cross
    case 'J': return 1u << 15;                 // square
    // analog sticks (pad.cpp turns bits 16-23 into stick positions): WASD left, TFGH right
    case 'W': return 1u << 16;
    case 'S': return 1u << 17;
    case 'A': return 1u << 18;
    case 'D': return 1u << 19;
    case 'T': return 1u << 20;
    case 'G': return 1u << 21;
    case 'F': return 1u << 22;
    case 'H': return 1u << 23;
    }
    return 0;
}

bool gs_gpu_active();
static LRESULT CALLBACK wndproc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
    case WM_KEYDOWN: if (w == VK_ESCAPE) { rt_log(LOG_INFO, "window: Esc pressed, exiting"); ExitProcess(0); } g_keys |= key_bit(w); return 0;
    case WM_KEYUP: g_keys &= ~key_bit(w); return 0;
    case WM_KILLFOCUS: g_keys = 0; return 0;   // releases happen elsewhere now
    case WM_CLOSE: rt_log(LOG_INFO, "window closed"); ExitProcess(0);
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(h, &ps);
        if (gs_gpu_active()) { EndPaint(h, &ps); return 0; }   // the GPU renderer presents directly
        RECT rc; GetClientRect(h, &rc);
        static const char *flt = getenv("NPPL_FILTER");
        static const bool nearest = flt && _stricmp(flt, "nearest") == 0;
        static const float sharpen = getenv("NPPL_SHARPEN") ? (float)atof(getenv("NPPL_SHARPEN")) : 0.6f;
        BITMAPINFO bi{};
        bi.bmiHeader.biSize = sizeof bi.bmiHeader;
        bi.bmiHeader.biPlanes = 1; bi.bmiHeader.biBitCount = 32; bi.bmiHeader.biCompression = BI_RGB;
        int cw = rc.right, ch = rc.bottom;
        if (nearest || cw <= 0 || ch <= 0 || cw > 4096 || ch > 4096) {
            std::lock_guard<std::mutex> lk(g_mx);
            bi.bmiHeader.biWidth = MAX_W; bi.bmiHeader.biHeight = -MAX_H;
            SetStretchBltMode(dc, COLORONCOLOR);
            StretchDIBits(dc, 0, 0, cw, ch, 0, MAX_H - g_fh, g_fw, g_fh, g_front.data(), &bi, DIB_RGB_COLORS, SRCCOPY);
        } else {
            // copy the frame out under the lock (the emulation thread swaps buffers each vblank),
            // then scale without holding it
            static std::vector<uint32_t> snap(MAX_W * MAX_H);
            int fw, fh; uint64_t gen;
            {
                std::lock_guard<std::mutex> lk(g_mx);
                fw = g_fw; fh = g_fh; gen = g_front_gen;
                if (gen != g_scaler.gen)
                    for (int y = 0; y < fh; y++) memcpy(&snap[(size_t)y * MAX_W], &g_front[(size_t)y * MAX_W], (size_t)fw * 4);
            }
            if (gen != g_scaler.gen || g_scaler.ow != cw || g_scaler.oh != ch) {
                upscale(g_scaler, snap.data(), fw, fh, cw, ch, sharpen);
                g_scaler.gen = gen;
            }
            bi.bmiHeader.biWidth = g_scaler.ow; bi.bmiHeader.biHeight = -g_scaler.oh;
            SetDIBitsToDevice(dc, 0, 0, g_scaler.ow, g_scaler.oh, 0, 0, 0, g_scaler.oh, g_scaler.out.data(), &bi, DIB_RGB_COLORS);
        }
        EndPaint(h, &ps);
        return 0;
    }
    }
    return DefWindowProcA(h, m, w, l);
}

static void ui_thread() {
    WNDCLASSA wc{};
    wc.lpfnWndProc = wndproc;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.lpszClassName = "nppl_pc";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassA(&wc);
    RECT r{0, 0, 960, 672};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    g_hwnd = CreateWindowA("nppl_pc", "NPPL Championship Paintball 2009 (PC)", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                           CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, nullptr, nullptr, wc.hInstance, nullptr);
    MSG msg;
    while (GetMessageA(&msg, nullptr, 0, 0) > 0) { TranslateMessage(&msg); DispatchMessageA(&msg); }
}

void display_init(bool headless) {
    if (headless) return;
    std::thread(ui_thread).detach();
}

// Called on every vblank from the emulation thread: the GS presents the frame in order with
// the drawing queued before it (on the GS thread when it has one).
void gs_vsync();
void gs_drain_stats(uint64_t &n, uint64_t &ns);
void display_vsync() {
    static auto t0 = std::chrono::steady_clock::now();
    if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(3)) {
        t0 = std::chrono::steady_clock::now();
        uint64_t n, ns; gs_drain_stats(n, ns);
        rt_log(LOG_INFO, "gs queue drains: %llu, %llu ms", (unsigned long long)n, (unsigned long long)(ns / 1000000));
    }
    gs_vsync();
}
bool gs_gpu_active();
bool gs_gpu_present();
HWND display_hwnd() { return g_hwnd; }
static std::atomic<int> g_shot_req{0};
bool display_shot_wanted() { return g_shot_req.load() != 0; }
// GPU renderer: a frame for screenshots (0x00RRGGBB)
void display_set_frame(const uint32_t *px, int w, int h, int stride) {
    w = std::min(w, MAX_W); h = std::min(h, MAX_H);
    {
        std::lock_guard<std::mutex> lk(g_mx);
        for (int y = 0; y < h; y++) memcpy(&g_front[(size_t)y * MAX_W], px + (size_t)y * stride, (size_t)w * 4);
        g_fw = w; g_fh = h; g_front_gen++;
    }
    g_shot_req = 0;
}
void display_present() {
    int w, h;
    g_frames++;
    {   // rate report: game frames (DISPFB flips) and vblanks per wall second
        extern uint64_t g_gs_flips;
        static uint64_t vbl, flips0;
        static auto t0 = std::chrono::steady_clock::now();
        vbl++;
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el >= 3.0) {
            std::string kernel_irq_rates(double el);   // kernel.cpp
            rt_log(LOG_INFO, "rate: %.1f fps (%.1f vblank/s) | irq %s", (g_gs_flips - flips0) / el, vbl / el,
                   kernel_irq_rates(el).c_str());
            vbl = 0; flips0 = g_gs_flips; t0 = std::chrono::steady_clock::now();
        }
    }
    if (gs_gpu_present()) return;
    if (!gs_read_display(g_back.data(), MAX_W, MAX_H, w, h)) return;
    {
        std::lock_guard<std::mutex> lk(g_mx);
        g_front.swap(g_back);
        g_fw = w; g_fh = h;
        g_front_gen++;
    }
    if (g_hwnd) InvalidateRect(g_hwnd, nullptr, FALSE);
}

// Debug: write the current display to a .bmp
bool display_dump_bmp(const char *path) {
    if (gs_gpu_active()) {   // ask the GPU renderer for the next presented frame
        g_shot_req = 1;
        for (int i = 0; i < 100 && g_shot_req; i++) Sleep(10);
    }
    std::lock_guard<std::mutex> lk(g_mx);
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    int w = g_fw, h = g_fh;
    uint32_t sz = 54 + w * h * 4;
    uint8_t hdr[54] = {'B', 'M'};
    memcpy(hdr + 2, &sz, 4); uint32_t off = 54; memcpy(hdr + 10, &off, 4);
    uint32_t hs = 40; memcpy(hdr + 14, &hs, 4); memcpy(hdr + 18, &w, 4); int nh = -h; memcpy(hdr + 22, &nh, 4);
    uint16_t planes = 1, bpp = 32; memcpy(hdr + 26, &planes, 2); memcpy(hdr + 28, &bpp, 2);
    fwrite(hdr, 1, 54, f);
    for (int y = 0; y < h; y++) fwrite(&g_front[y * MAX_W], 4, w, f);
    fclose(f);
    if (const char *up = getenv("NPPL_SHOTUP")) {   // debug: also save the window upscaler's output at <scale>x
        float k = (float)atof(up); if (k < 1) k = 2;
        static Scaler sc;
        int ow = (int)(w * k), oh = (int)(h * k);
        upscale(sc, g_front.data(), w, h, ow, oh, getenv("NPPL_SHARPEN") ? (float)atof(getenv("NPPL_SHARPEN")) : 0.6f);
        std::string p2 = std::string(path);
        p2 = p2.substr(0, p2.size() - 4) + "_up.bmp";
        if (FILE *g = fopen(p2.c_str(), "wb")) {
            uint32_t sz2 = 54 + ow * oh * 4; memcpy(hdr + 2, &sz2, 4); memcpy(hdr + 18, &ow, 4); int nh2 = -oh; memcpy(hdr + 22, &nh2, 4);
            fwrite(hdr, 1, 54, g);
            fwrite(sc.out.data(), 4, (size_t)ow * oh, g);
            fclose(g);
        }
    }
    return true;
}
