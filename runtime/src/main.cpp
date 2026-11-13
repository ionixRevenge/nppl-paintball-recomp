// Entry point: load the user's executable from iso/ and run the recompiled code.
#include <mutex>
#include "rt_internal.h"
#include "hw.h"
#include "sif.h"
#include "iso.h"
#define NOMINMAX
#include <windows.h>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include <vector>

int g_log_level = LOG_INFO;
static FILE *g_logf;

void rt_log(int level, const char *fmt, ...) {
    if (level > g_log_level) return;
    static const char *tag[] = {"ERR ", "WARN", "INFO", "TRC "};
    char buf[1024];
    va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    fprintf(stderr, "[%s] %s\n", tag[level], buf);
    if (g_logf) { fprintf(g_logf, "[%s] %s\n", tag[level], buf); fflush(g_logf); }
}

bool rt_once(uint64_t key) {
    static std::mutex mx;
    static std::unordered_set<uint64_t> seen;
    std::lock_guard<std::mutex> lk(mx);
    return seen.insert(key).second;
}

[[noreturn]] void rt_exit(int code) {
    rt_log(LOG_INFO, "exit(%d)", code);
    if (g_logf) fclose(g_logf);
    std::exit(code);
}

static R5900 g_main;
static auto g_start = std::chrono::steady_clock::now();
// Emulated time (ns since boot). Normally wall-clock time; NPPL_TIMESCALE=<f> makes it run at
// that fraction of real time from vblank NPPL_TIMESCALE_AT on (debug: see how the game behaves
// when it gets its intended frame time). Monotonic across scale changes.
extern uint32_t g_vsync_count;
double rt_timescale() {
    static const double s = getenv("NPPL_TIMESCALE") ? atof(getenv("NPPL_TIMESCALE")) : 1.0;
    static const uint32_t at = getenv("NPPL_TIMESCALE_AT") ? (uint32_t)atoi(getenv("NPPL_TIMESCALE_AT")) : 0;
    return (s > 0 && g_vsync_count >= at) ? s : 1.0;
}
uint64_t rt_emu_ns() {
    static std::mutex mx;
    static auto last = std::chrono::steady_clock::now();
    static double acc;
    std::lock_guard<std::mutex> lk(mx);
    auto now = std::chrono::steady_clock::now();
    acc += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now - last).count() * rt_timescale();
    last = now;
    return (uint64_t)acc;
}

extern "C" {

bool rt_has_func(uint32_t addr) {
    unsigned lo = 0, hi = rt_func_count;
    while (lo < hi) {
        unsigned mid = (lo + hi) / 2;
        if (rt_func_table[mid].addr < addr) lo = mid + 1; else hi = mid;
    }
    return lo < rt_func_count && rt_func_table[lo].addr == addr;
}

void rt_call(R5900 *c, uint32_t addr) {
    // binary search the generated function table
    unsigned lo = 0, hi = rt_func_count;
    while (lo < hi) {
        unsigned mid = (lo + hi) / 2;
        if (rt_func_table[mid].addr < addr) lo = mid + 1; else hi = mid;
    }
    if (lo < rt_func_count && rt_func_table[lo].addr == addr) { rt_func_table[lo].fn(c); return; }
    rt_log(LOG_ERR, "call to unknown address %08X (ra=%08X, thread %d)", addr, (uint32_t)c->r[31].ud[0], c->thread_id);
    rt_exit(2);
}

void rt_bad_entry(R5900 *c, uint32_t addr) {
    rt_log(LOG_ERR, "bad entry %08X (thread %d)", addr, c->thread_id);
    rt_exit(2);
}

// Guest backtrace: saved return addresses on the guest stack that point just
// after a jal/jalr in game code.
void rt_backtrace(R5900 *c) {
    uint32_t sp = (uint32_t)c->r[29].ud[0];
    int found = 0;
    for (uint32_t a = sp; a < sp + 0x1000 && a < RAM_SIZE && found < 16; a += 4) {
        uint32_t v = LW(a);
        if (v < 0x100008 || v >= 0x76C5E0 || (v & 3)) continue;
        uint32_t call = LW(v - 8);
        bool is_jal = (call >> 26) == 3, is_jalr = (call & 0xFC00003F) == 0x00000009;
        if (!is_jal && !is_jalr) continue;
        rt_log(LOG_ERR, "  #%d return to %08X (call at %08X%s%08X) [sp+%X]", found++, v, v - 8,
               is_jal ? " -> " : " jalr ", is_jal ? (((v - 8) & 0xF0000000) | ((call & 0x3FFFFFF) << 2)) : 0, a - sp);
    }
}

void rt_break(R5900 *c, uint32_t pc) {
    rt_log(LOG_ERR, "break at %08X (ra=%08X)", pc, (uint32_t)c->r[31].ud[0]);
    rt_backtrace(c);
    rt_exit(3);
}

void rt_invalid(R5900 *c, uint32_t pc, uint32_t w) {
    rt_log(LOG_ERR, "executed invalid instruction %08X at %08X (ra=%08X)", w, pc, (uint32_t)c->r[31].ud[0]);
    rt_exit(3);
}

void rt_unimpl(R5900 *c, uint32_t pc, const char *op) {
    if (rt_once(0xB00000000ull | pc)) rt_log(LOG_WARN, "unimplemented %s at %08X", op, pc);
    (void)c;
}

uint32_t rt_mfc0(R5900 *c, int r) {
    if (r == 9) {  // Count: 294.912 MHz
        return (uint32_t)((double)rt_emu_ns() * 0.294912);
    }
    if (r == 15) return 0x2E20;  // PRId: R5900
    return c->cop0[r];
}
void rt_mtc0(R5900 *c, int r, uint32_t v) { c->cop0[r] = v; }
uint32_t rt_mfperf(R5900 *c, int sel) { return rt_mfc0(c, 9); (void)sel; }
void rt_mtperf(R5900 *, int, uint32_t) {}

} // extern "C"

R5900 *kernel_current_ctx();

// Debug aid: periodically report where the current guest thread is,
// plus any guest words requested with -m ADDR (repeatable).
static std::vector<uint32_t> g_watch_addrs;
std::string kernel_debug_state();
std::string dma_debug_state();
void display_init(bool headless);
bool display_dump_bmp(const char *path);
void gs_init();
void vu_init();
static bool g_shots;
extern uint32_t g_pad_script;
extern bool g_audio_mute;
extern const char *g_audio_wav;
extern bool g_movies;
void pad_set_script(const char *spec);
void profiler_start();
// Debug: NPPL_WATCHW="addr[,addr...]" (up to 4 guest addresses, 4-byte aligned) puts CPU data
// breakpoints on writes to them once NPPL_WATCHW_AT vblanks have passed (default 0). Guest code
// runs as fibers on the main thread, so the debug registers there catch every guest store. Each
// hit is counted by host code address (symbolised to the generated f_XXXXXXXX) and the guest
// $ra; the watchdog prints and resets the tally every 3 s. NPPL_WATCHW_LOG=<n> also logs the
// first n individual hits with the value written.
#include <dbghelp.h>
#include <map>
#include <algorithm>
static HANDLE g_main_thread;
static uint32_t g_ww_addr[4];
static int g_ww_n, g_ww_log;
static uint32_t g_ww_at;
static bool g_ww_armed, g_ww_find;
static float g_ww_fx, g_ww_fz;
static std::mutex g_ww_mu;
static FILE *g_ww_csv;
static uint32_t g_ww_base[4];   // body base per watch (body mode), for the CSV
static std::map<std::pair<uint64_t, uint32_t>, uint32_t> g_ww_hits;   // (host rip, guest ra) -> count

static std::string host_sym(uint64_t a) {
    static bool init = [] { SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS); SymInitialize(GetCurrentProcess(), nullptr, TRUE); return true; }();
    (void)init;
    alignas(SYMBOL_INFO) char buf[sizeof(SYMBOL_INFO) + 256];
    SYMBOL_INFO *si = (SYMBOL_INFO *)buf;
    si->SizeOfStruct = sizeof(SYMBOL_INFO); si->MaxNameLen = 255;
    DWORD64 disp = 0;
    char out[300];
    if (SymFromAddr(GetCurrentProcess(), a, &disp, si)) snprintf(out, sizeof out, "%s+%llX", si->Name, (unsigned long long)disp);
    else snprintf(out, sizeof out, "%llX", (unsigned long long)a);
    return out;
}

static LONG WINAPI ww_handler(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP) return EXCEPTION_CONTINUE_SEARCH;
    DWORD64 dr6 = ep->ContextRecord->Dr6;
    if (!(dr6 & 0xF)) return EXCEPTION_CONTINUE_SEARCH;
    R5900 *c = kernel_current_ctx();
    uint32_t ra = c ? (uint32_t)c->r[31].ud[0] : 0;
    uint64_t rip = (uint64_t)ep->ExceptionRecord->ExceptionAddress;
    {
        std::lock_guard<std::mutex> lk(g_ww_mu);
        g_ww_hits[{rip, ra}]++;
    }
    // NPPL_WATCHW_CSV=<file>: every hit as vblank,addr,rip,ra,v[0],v[2],v[-0x90],v[-0x88]
    // (for a body velocity watch: vel.x, vel.z, pos.x, pos.z)
    FILE *csv = g_ww_csv;
    if (csv) {
        for (int k = 0; k < g_ww_n; k++)
            if (dr6 & (1u << k)) {
                uint32_t b = g_ww_base[k] ? g_ww_base[k] : g_ww_addr[k] - 0x90;
                const float *v = (const float *)(g_ram + b + 0x90), *t = (const float *)(g_ram + b);
                fprintf(csv, "%u,%08X,%llX,%08X,%.4f,%.4f,%.4f,%.4f\n", g_vsync_count, g_ww_addr[k],
                        (unsigned long long)rip, ra, v[0], v[2], t[0], t[2]);
            }
    }
    if (g_ww_log > 0) {
        for (int k = 0; k < g_ww_n; k++)
            if (dr6 & (1u << k)) {
                g_ww_log--;
                uint32_t v = LW(g_ww_addr[k]); float f; memcpy(&f, &v, 4);
                rt_log(LOG_INFO, "watchw: vbl %u [%08X]=%08X (%g) rip %llX ra %08X", g_vsync_count, g_ww_addr[k], v, f,
                       (unsigned long long)rip, ra);
            }
    }
    ep->ContextRecord->Dr6 = 0;
    return EXCEPTION_CONTINUE_EXECUTION;
}

static void ww_init() {
    const char *s = getenv("NPPL_WATCHW");
    if (const char *f = getenv("NPPL_WATCHW_FIND")) {   // "x,z": watch the first float pair x @+0, z @+8
        sscanf(f, "%f,%f", &g_ww_fx, &g_ww_fz);
        g_ww_find = true;
        s = "";
    }
    if (!s) return;
    for (const char *p = s; *p && g_ww_n < 4;) {
        char *e; uint32_t a = (uint32_t)strtoul(p, &e, 16);
        if (e == p) break;
        g_ww_addr[g_ww_n++] = a & 0x1FFFFFC;
        p = *e ? e + 1 : e;
    }
    if (const char *at = getenv("NPPL_WATCHW_AT")) g_ww_at = (uint32_t)strtoul(at, nullptr, 0);
    if (const char *lg = getenv("NPPL_WATCHW_LOG")) g_ww_log = atoi(lg);
    if (const char *cf = getenv("NPPL_WATCHW_CSV")) g_ww_csv = fopen(cf, "w");
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &g_main_thread, 0, FALSE, DUPLICATE_SAME_ACCESS);
    AddVectoredExceptionHandler(1, ww_handler);
}

static void ww_poll() {   // watchdog thread
    if (g_ww_find && g_ww_armed && GetFileAttributesA("watchw_refind.flag") != INVALID_FILE_ATTRIBUTES) {
        DeleteFileA("watchw_refind.flag");   // touch watchw_refind.flag: drop the breakpoints and search again
        CONTEXT ctx = {};
        ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;
        SuspendThread(g_main_thread);
        if (GetThreadContext(g_main_thread, &ctx)) { ctx.Dr7 = 0; SetThreadContext(g_main_thread, &ctx); }
        ResumeThread(g_main_thread);
        g_ww_armed = false; g_ww_n = 0;
        rt_log(LOG_INFO, "watchw: re-searching at vblank %u", g_vsync_count);
    }
    if (g_ww_find && !g_ww_n && g_vsync_count >= g_ww_at) {
        // NPPL character: position (x, y, z) at P, then its box max (x+0.8, y+1.8, z+0.8) at P+12.
        // "0,0" finds the first 4 characters; otherwise only the one at x,z.
        const float *f = (const float *)g_ram;
        uint32_t found[8]; int nf = 0;
        for (uint32_t i = 0x100000 / 4; i + 6 < (32u << 20) / 4 && nf < 8; i++) {
            float x = f[i], y = f[i + 1], z = f[i + 2];
            if (!(y > 0.2f && y < 5.0f) || !(std::fabs(x) > 1.0f && std::fabs(x) < 500.0f) ||
                !(std::fabs(z) > 1.0f && std::fabs(z) < 500.0f)) continue;
            bool fin = true;
            for (int k = 0; k < 6; k++) fin = fin && std::isfinite(f[i + k]);
            if (!fin || f[i + 1] == f[i] || f[i + 2] == f[i]) continue;
            if (std::fabs(f[i + 3] - x - 0.8f) > 0.002f || std::fabs(f[i + 4] - y - 1.8f) > 0.002f ||
                std::fabs(f[i + 5] - z - 0.8f) > 0.002f) continue;
            if ((g_ww_fx || g_ww_fz) && (std::fabs(x - g_ww_fx) > 1.0f || std::fabs(z - g_ww_fz) > 1.0f)) continue;
            bool dup = false;   // the same position is mirrored in other structs: keep one per character
            for (int k = 0; k < nf; k++) { const float *o = f + found[k] / 4; if (o[0] == x && o[2] == z) dup = true; }
            if (dup) continue;
            found[nf++] = i * 4;
            rt_log(LOG_INFO, "watchw: character at %08X pos %.3f %.3f %.3f", i * 4, x, y, z);
        }
        // NPPL_WATCHW_BODY=1: watch the Havok rigid bodies of the first two characters instead:
        // hkpMotion translation = (x, y + ~0.92, z) of the character position (box corner + 0x24),
        // with unit-w quaternions at +0x3C/+0x4C; linear velocity x/z at +0x90/+0x98.
        if (nf >= 3 && getenv("NPPL_WATCHW_BODY")) {
            int nb = 0;
            uint32_t body[8];
            for (int k = 0; k < nf && nb < 8; k++) {
                const float *p = f + (found[k] + 0x24) / 4;
                for (uint32_t i = 0x100000 / 4; i + 0x14 < (32u << 20) / 4; i += 4)
                    if (f[i] == p[0] && f[i + 2] == p[2] && f[i + 1] - p[1] > 0.5f && f[i + 1] - p[1] < 1.5f &&
                        f[i + 0x3C / 4] == 1.0f && f[i + 0x4C / 4] == 1.0f) {
                        body[nb++] = i * 4;
                        rt_log(LOG_INFO, "watchw: body of character %08X at %08X vel %.3f %.3f %.3f", found[k], i * 4,
                               f[i + 0x24], f[i + 0x25], f[i + 0x26]);
                        break;
                    }
            }
            if (nb >= 2) {   // x velocity of up to 4 bodies, moving (alive) ones first
                // nearest to the human player's usual spawn (51.77, 56.41) first: player, then teammates
                auto d2 = [&](uint32_t b) { const float *q = f + b / 4; return (q[0] - 51.77f) * (q[0] - 51.77f) + (q[2] - 56.41f) * (q[2] - 56.41f); };
                std::sort(body, body + nb, [&](uint32_t a, uint32_t b) { return d2(a) < d2(b); });
                int n = 0;
                // NPPL_WATCHW_BODYOFF: which body field to watch (0x90 = vel.x, 0 = translation.x)
                static const uint32_t boff = getenv("NPPL_WATCHW_BODYOFF") ? (uint32_t)strtoul(getenv("NPPL_WATCHW_BODYOFF"), nullptr, 0) : 0x90;
                for (int k = 0; k < nb && n < 4; k++) { g_ww_base[n] = body[k]; g_ww_addr[n++] = body[k] + boff; }
                for (int k = 0; k < n; k++) rt_log(LOG_INFO, "watchw: watching vel.x at %08X", g_ww_addr[k]);
                g_ww_n = n;
            }
            nf = 0;   // don't fall through to the position watch
        }
        if (nf >= 3 || ((g_ww_fx || g_ww_fz) && nf)) {
            // NPPL_WATCHW_OFF: byte offset from the box corner to watch (0x24 = the position itself)
            static const int off = getenv("NPPL_WATCHW_OFF") ? (int)strtol(getenv("NPPL_WATCHW_OFF"), nullptr, 0) : 0;
            if (nf > 4) nf = 4;
            for (int k = 0; k < nf; k++) g_ww_addr[k] = found[k] + off;
            g_ww_n = nf;
        }
    }
    if (!g_ww_n) return;
    if (!g_ww_armed && g_vsync_count >= g_ww_at) {
        CONTEXT ctx = {};
        ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;
        SuspendThread(g_main_thread);
        if (GetThreadContext(g_main_thread, &ctx)) {
            DWORD64 *dr[4] = {&ctx.Dr0, &ctx.Dr1, &ctx.Dr2, &ctx.Dr3};
            ctx.Dr7 = 0;
            for (int k = 0; k < g_ww_n; k++) {
                *dr[k] = (DWORD64)(g_ram + g_ww_addr[k]);
                ctx.Dr7 |= (1ull << (k * 2)) | (1ull << (16 + k * 4)) | (3ull << (18 + k * 4));   // local enable, write, 4 bytes
            }
            SetThreadContext(g_main_thread, &ctx);
            g_ww_armed = true;
            rt_log(LOG_INFO, "watchw: armed %d address(es) at vblank %u", g_ww_n, g_vsync_count);
        }
        ResumeThread(g_main_thread);
        return;
    }
    if (!g_ww_armed) return;
    if (g_ww_csv) fflush(g_ww_csv);
    if (GetFileAttributesA("watchw.flag") != INVALID_FILE_ATTRIBUTES) {   // touch watchw.flag: log the next 300 hits
        DeleteFileA("watchw.flag");
        g_ww_log = 300;
        rt_log(LOG_INFO, "watchw: logging burst at vblank %u", g_vsync_count);
    }
    std::map<std::pair<uint64_t, uint32_t>, uint32_t> h;
    { std::lock_guard<std::mutex> lk(g_ww_mu); h.swap(g_ww_hits); }
    for (auto &e : h)
        rt_log(LOG_INFO, "watchw: %6u x %s ra %08X", e.second, host_sym(e.first.first).c_str(), e.first.second);
}

static void watchdog() {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        ww_poll();
        R5900 *c = kernel_current_ctx();
        if (c) rt_log(LOG_INFO, "watchdog: thread %d at %08X (ra=%08X sp=%08X)", c->thread_id, c->pc,
                      (uint32_t)c->r[31].ud[0], (uint32_t)c->r[29].ud[0]);
        if (!g_watch_addrs.empty()) {
            std::string s;
            char b[32];
            for (uint32_t a : g_watch_addrs) { snprintf(b, sizeof b, " [%08X]=%08X", a, LW(a)); s += b; }
            rt_log(LOG_INFO, "watch:%s | %s", s.c_str(), kernel_debug_state().c_str());
        }
        rt_log(LOG_INFO, "gfx: %s", dma_debug_state().c_str());
        if (g_shots) {
            static int n;
            char name[64]; snprintf(name, sizeof name, "shot_%03d.bmp", n++);
            display_dump_bmp(name);
            static const bool keep = getenv("NPPL_RAMDUMPS") != nullptr;   // debug: one RAM snapshot per screenshot
            char rn[64]; snprintf(rn, sizeof rn, keep ? "ram_%03d.bin" : "ee_ram.bin", n - 1);
            if (FILE *f = fopen(rn, "wb")) { fwrite(g_ram, 1, 32 << 20, f); fclose(f); }
        }
    }
}

// Load the game executable straight from the user's disc image.
static bool load_elf(const std::string &disc_path, uint32_t &entry) {
    IsoEntry e;
    if (!iso_lookup(disc_path, e)) return false;
    std::vector<uint8_t> d(e.size);
    if (!iso_read_bytes(e, 0, d.data(), d.size())) return false;
    if (d.size() < 0x34 || memcmp(d.data(), "" "ELF", 4)) return false;
    auto u32 = [&](size_t o) { uint32_t v; memcpy(&v, &d[o], 4); return v; };
    auto u16 = [&](size_t o) { uint16_t v; memcpy(&v, &d[o], 2); return v; };
    entry = u32(0x18);
    uint32_t phoff = u32(0x1C), phnum = u16(0x2C);
    for (uint32_t i = 0; i < phnum; i++) {
        size_t p = phoff + i * 32;
        uint32_t type = u32(p), off = u32(p + 4), va = u32(p + 8), fsz = u32(p + 16), msz = u32(p + 20);
        if (type != 1 || !msz) continue;
        memcpy(g_ram + va, &d[off], fsz);
        memset(g_ram + va + fsz, 0, msz - fsz);
        rt_log(LOG_INFO, "loaded segment %08X..%08X", va, va + msz);
    }
    return true;
}

// Log host crashes with the guest context, so a fault can be traced back to recompiled code.
static LONG WINAPI crash_handler(EXCEPTION_POINTERS *ep) {
    static bool once;
    if (once) return EXCEPTION_CONTINUE_SEARCH;
    once = true;
    const EXCEPTION_RECORD *er = ep->ExceptionRecord;
    HMODULE mod = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)er->ExceptionAddress, &mod);
    rt_log(LOG_ERR, "host crash: code %08lX at %p (module+%llX) access %s %p", er->ExceptionCode, er->ExceptionAddress,
           (unsigned long long)((uintptr_t)er->ExceptionAddress - (uintptr_t)mod),
           er->NumberParameters >= 2 ? (er->ExceptionInformation[0] ? "write" : "read") : "?",
           er->NumberParameters >= 2 ? (void *)er->ExceptionInformation[1] : nullptr);
    if (R5900 *c = kernel_current_ctx())
        rt_log(LOG_ERR, "  guest thread %d pc %08X ra %08X sp %08X", c->thread_id, c->pc,
               (uint32_t)c->r[31].ud[0], (uint32_t)c->r[29].ud[0]);
    if (g_logf) fflush(g_logf);
    return EXCEPTION_CONTINUE_SEARCH;
}

// Ask hybrid-graphics laptops for the discrete GPU (NVIDIA Optimus / AMD PowerXpress).
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 1;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

int main(int argc, char **argv) {
    SetUnhandledExceptionFilter(crash_handler);
    std::string disc = "game_iso";
    bool headless = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-v") g_log_level = LOG_TRACE;
        else if (a == "-q") g_log_level = LOG_WARN;
        else if (a == "-w") std::thread(watchdog).detach();
        else if (a == "-s") g_shots = true;
        else if (a == "-press" && i + 1 < argc) g_pad_script = (uint32_t)strtoul(argv[++i], nullptr, 0);  // pulse these pad bits
        else if (a == "-headless") headless = true;
        else if (a == "-mute") g_audio_mute = true;
        else if (a == "-gpu") _putenv("NPPL_GPU=1");
        else if (a == "-sw") _putenv("NPPL_GPU=0");
        else if (a == "-texdump") _putenv("NPPL_TEXDUMP=1");
        else if (a == "-texscale" && i + 1 < argc) _putenv(("NPPL_TEXSCALE=" + std::string(argv[++i])).c_str());
        else if (a == "-scale" && i + 1 < argc) _putenv(("NPPL_SCALE=" + std::string(argv[++i])).c_str());
        else if (a == "-movies") g_movies = true;
        else if (a == "-prof") profiler_start();   // sampling profiler, reports every 20 s
        else if (a == "-input" && i + 1 < argc) pad_set_script(argv[++i]);   // scripted pad input   // try to play .PSS movies (needs IPU emulation)
        else if (a == "-wav" && i + 1 < argc) g_audio_wav = argv[++i];   // capture mixed audio
        else if (a == "-m" && i + 1 < argc) g_watch_addrs.push_back((uint32_t)strtoul(argv[++i], nullptr, 16));
        else disc = a;
    }
    g_logf = fopen("ps2rt.log", "w");
    g_ram = (uint8_t *)calloc(1, RAM_SIZE);
    hw_init();
    sif_init();
    gs_init();
    vu_init();
    display_init(headless);

    if (!iso_open(disc)) {
        rt_log(LOG_ERR, "Put your own NPPL Championship Paintball 2009 (USA) ISO in game_iso/ "
                        "or pass its path: nppl_pc <path-to-iso-or-folder>");
        return 1;
    }
    uint32_t entry;
    if (!load_elf("cdrom0:\SLUS_218.55;1", entry)) {
        rt_log(LOG_ERR, "SLUS_218.55 not found on the disc: is this the USA release?");
        return 1;
    }
    memset(&g_main, 0, sizeof g_main);
    g_main.vf[0] = VF0;
    kernel_init(&g_main);
    ww_init();
    rt_log(LOG_INFO, "starting at %08X", entry);
    rt_call(&g_main, entry);
    rt_log(LOG_INFO, "entry returned");
    rt_exit(0);
}
