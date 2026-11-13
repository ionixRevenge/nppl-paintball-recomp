// VU1 interpreter benchmark / regression check: replays program runs captured with
// NPPL_VUSNAP=<n> (vusnap.bin) and reports speed plus a checksum of every result
// (registers, data memory and XGKICKed packets), so interpreter changes can be verified.
#include "rt_internal.h"
#include "vu.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>
#include <atomic>
#include <thread>
#include <map>
#include <string>
#include <algorithm>
#pragma comment(lib, "dbghelp.lib")
static std::atomic<bool> g_stop;
static std::map<uint64_t, uint32_t> g_hist;
static uint64_t g_ns;
static void sampler(HANDLE t) {
    timeBeginPeriod(1);
    while (!g_stop) {
        Sleep(1);
        if (SuspendThread(t) == (DWORD)-1) continue;
        CONTEXT c{}; c.ContextFlags = CONTEXT_CONTROL;
        if (GetThreadContext(t, &c)) { g_hist[c.Rip]++; g_ns++; }
        ResumeThread(t);
    }
}
static void prof_report() {
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_LOAD_LINES);
    SymInitialize(proc, nullptr, TRUE);
    std::map<std::string, uint32_t> by;
    for (auto &h : g_hist) {
        IMAGEHLP_LINE64 ln{}; ln.SizeOfStruct = sizeof ln; DWORD d = 0;
        alignas(SYMBOL_INFO) char buf[sizeof(SYMBOL_INFO) + 256];
        SYMBOL_INFO *si = (SYMBOL_INFO *)buf; memset(si, 0, sizeof(SYMBOL_INFO)); si->SizeOfStruct = sizeof(SYMBOL_INFO); si->MaxNameLen = 255;
        DWORD64 d2 = 0;
        std::string nm = SymFromAddr(proc, h.first, &d2, si) ? si->Name : "?";
        if (nm == "_NLG_Return2" || nm == "?") { char t[64]; snprintf(t, sizeof t, "+%llx", (unsigned long long)(nm == "?" ? h.first : d2) & ~0xFull); nm += t; }
        char key[400];
        if (SymGetLineFromAddr64(proc, h.first, &d, &ln)) {
            const char *f = strrchr(ln.FileName, 92); f = f ? f + 1 : ln.FileName;
            snprintf(key, sizeof key, "%s:%lu (%s)", f, ln.LineNumber, nm.c_str());
        } else snprintf(key, sizeof key, "%s", nm.c_str());
        by[key] += h.second;
    }
    std::vector<std::pair<uint32_t, std::string>> v;
    for (auto &b : by) v.push_back({b.second, b.first});
    std::sort(v.rbegin(), v.rend());
    for (size_t i = 0; i < v.size() && i < 45; i++) printf("%5.1f%% %s\n", 100.0 * v[i].first / g_ns, v[i].second.c_str());
}

uint8_t g_vu0_mem[0x1000], g_vu0_micro[0x1000], g_vu1_mem[0x4000], g_vu1_micro[0x4000];
uint64_t g_prof_vu_ns, g_vu_instr_dummy;
static uint64_t g_sum = 1469598103934665603ull;
static void mix(const void *p, size_t n) {   // n is a multiple of 8
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, b + i, 8); g_sum = (g_sum ^ w) * 0x9E3779B97F4A7C15ull; g_sum ^= g_sum >> 31; }
}
void rt_log(int, const char *fmt, ...) { (void)fmt; }
bool rt_once(uint64_t) { return false; }
int vu0_flags_legacy = getenv("NPPL_VUFLAGS_OLD") && atoi(getenv("NPPL_VUFLAGS_OLD")) != 0;
uint64_t rt_emu_ns() { return 0; }
double rt_timescale() { return 1.0; }
void vu1_xgkick_hook(const uint8_t *base, uint32_t addr, uint32_t size) {
    static const bool ns = getenv("VUB_NOSUM") != nullptr; if (ns) return;
    // checksum the GIF packet: walk tags until EOP
    for (int guard = 0; guard < 64; guard++) {
        uint64_t tag; memcpy(&tag, base + (addr % size), 8);
        uint32_t nloop = tag & 0x7FFF, flg = (tag >> 58) & 3, nreg = (tag >> 60) & 15; if (!nreg) nreg = 16;
        uint32_t qw = flg == 0 ? nloop * nreg : flg == 1 ? (nloop * nreg + 1) / 2 : nloop;
        for (uint32_t i = 0; i <= qw; i++) mix(base + ((addr + i * 16) % size), 16);
        addr += (qw + 1) * 16;
        if ((tag >> 15) & 1) break;
    }
}
extern uint64_t g_vu_instr;

struct Snap { uint32_t pc; VuState st; R5900 regs; std::vector<uint8_t> micro, mem; };

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "vusnap.bin";
    int iters = argc > 2 ? atoi(argv[2]) : 20;
    FILE *f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    std::vector<Snap> snaps;
    for (;;) {
        Snap s; s.micro.resize(0x4000); s.mem.resize(0x4000);
        if (fread(&s.pc, 4, 1, f) != 1) break;
        fread(&s.st, sizeof s.st, 1, f); fread(&s.regs, sizeof s.regs, 1, f);
        fread(s.micro.data(), 1, 0x4000, f); fread(s.mem.data(), 1, 0x4000, f);
        snaps.push_back(std::move(s));
    }
    fclose(f);
    vu_init();
    static R5900 regs;
    uint64_t instr0 = g_vu_instr;
    double setup_ms = 0;
    std::thread samp;
    if (getenv("VUB_PROF")) {
        HANDLE h; DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &h, THREAD_ALL_ACCESS, FALSE, 0);
        samp = std::thread(sampler, h);
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; it++) {
        g_sum = 1469598103934665603ull;
        for (const Snap &s : snaps) {
            VuState &vu = g_vu1;
            uint8_t *mem = vu.mem, *micro = vu.micro;
            vu = s.st; vu.mem = mem; vu.micro = micro; vu.regs = &regs;
            regs = s.regs;
            if (memcmp(micro, s.micro.data(), 0x4000)) { memcpy(micro, s.micro.data(), 0x4000); vu_micro_dirty(1); } memcpy(mem, s.mem.data(), 0x4000);
            vu_execute(vu, s.pc);
            static const bool nosum = getenv("VUB_NOSUM") != nullptr;
            if (!nosum) { mix(regs.vf, sizeof regs.vf); mix(regs.vi, 16 * 4); mix(&regs.vacc, 16); mix(mem, 0x4000); }
        }
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    uint64_t n = g_vu_instr - instr0;
    printf("%zu runs x %d: %llu instr, %.1f ms, %.2f ns/instr, checksum %016llx\n", snaps.size(), iters,
           (unsigned long long)n, ms, ms * 1e6 / (double)n, (unsigned long long)g_sum);
    if (samp.joinable()) { g_stop = true; samp.join(); prof_report(); }
    return 0;
}
