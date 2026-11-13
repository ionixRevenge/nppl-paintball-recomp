// Sampling profiler (-prof): samples the game thread's instruction pointer every ~1 ms and
// periodically logs the hottest functions, resolved through the PDB (recompiled functions
// show up by their f_XXXXXXXX public symbol).
#include "rt_internal.h"
#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>
#include <algorithm>
#include <chrono>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <mutex>
static HANDLE g_target;
static std::mutex g_extra_mx;
static std::vector<HANDLE> g_extra;   // rasterizer threads, reported as a separate group
static bool g_prof_on;
void profiler_add_thread() {
    HANDLE h;
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &h,
                    THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0);
    std::lock_guard<std::mutex> lk(g_extra_mx);
    g_extra.push_back(h);
}

static void report(std::unordered_map<uint64_t, uint32_t> &hist, uint64_t total, const char *tag) {
    HANDLE proc = GetCurrentProcess();
    std::map<std::string, uint32_t> by_func;
    alignas(SYMBOL_INFO) char buf[sizeof(SYMBOL_INFO) + 256];
    for (auto &h : hist) {
        SYMBOL_INFO *si = (SYMBOL_INFO *)buf;
        memset(si, 0, sizeof(SYMBOL_INFO));
        si->SizeOfStruct = sizeof(SYMBOL_INFO); si->MaxNameLen = 255;
        DWORD64 disp = 0;
        std::string name = SymFromAddr(proc, h.first, &disp, si) ? si->Name : "?";
        static const bool lines = getenv("NPPL_PROFLINES") != nullptr;
        if (lines) {
            IMAGEHLP_LINE64 ln{}; ln.SizeOfStruct = sizeof ln; DWORD d2 = 0;
            if (SymGetLineFromAddr64(proc, h.first, &d2, &ln)) {
                const char *f = strrchr(ln.FileName, 92); f = f ? f + 1 : ln.FileName;
                name += " " + std::string(f) + ":" + std::to_string(ln.LineNumber);
            }
        }
        by_func[name] += h.second;
    }
    std::vector<std::pair<uint32_t, std::string>> v;
    for (auto &f : by_func) v.push_back({f.second, f.first});
    std::sort(v.rbegin(), v.rend());
    rt_log(LOG_INFO, "prof%s: %llu samples", tag, (unsigned long long)total);
    for (size_t i = 0; i < v.size() && i < (getenv("NPPL_PROFLINES") ? 60u : 25u); i++)
        rt_log(LOG_INFO, "prof%s: %5.1f%%  %s", tag, 100.0 * v[i].first / total, v[i].second.c_str());
}

static void sampler() {
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    std::unordered_map<uint64_t, uint32_t> hist, rhist;
    uint64_t total = 0, rtotal = 0;
    auto last = std::chrono::steady_clock::now();
    timeBeginPeriod(1);
    for (;;) {
        Sleep(1);
        if (SuspendThread(g_target) == (DWORD)-1) continue;
        static const bool stack = getenv("NPPL_PROFSTACK") != nullptr;
        CONTEXT ctx{}; ctx.ContextFlags = stack ? CONTEXT_FULL : CONTEXT_CONTROL;
        if (GetThreadContext(g_target, &ctx)) {
            uint64_t pc = ctx.Rip;
            // NPPL_PROFSTACK: attribute samples outside the exe (system waits) to the exe frame that called them
            static uint64_t lo, hi;
            if (!lo) { HMODULE m = GetModuleHandleA(nullptr); auto *dos = (IMAGE_DOS_HEADER *)m; auto *nt = (IMAGE_NT_HEADERS *)((uint8_t *)m + dos->e_lfanew); lo = (uint64_t)m; hi = lo + nt->OptionalHeader.SizeOfImage; }
            if (stack && (pc < lo || pc >= hi)) {
                STACKFRAME64 sf{}; sf.AddrPC.Offset = ctx.Rip; sf.AddrPC.Mode = AddrModeFlat;
                sf.AddrFrame.Offset = ctx.Rbp; sf.AddrFrame.Mode = AddrModeFlat; sf.AddrStack.Offset = ctx.Rsp; sf.AddrStack.Mode = AddrModeFlat;
                CONTEXT c2 = ctx;
                for (int i = 0; i < 24; i++) {
                    if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, GetCurrentProcess(), g_target, &sf, &c2, nullptr, SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) break;
                    uint64_t a = sf.AddrPC.Offset;
                    if (a >= lo && a < hi) { pc = a; break; }
                }
            }
            hist[pc]++; total++;
        }
        ResumeThread(g_target);
        {
            std::lock_guard<std::mutex> lk(g_extra_mx);
            for (HANDLE h : g_extra) {
                if (SuspendThread(h) == (DWORD)-1) continue;
                CONTEXT c2{}; c2.ContextFlags = CONTEXT_CONTROL;
                if (GetThreadContext(h, &c2)) { rhist[c2.Rip]++; rtotal++; }
                ResumeThread(h);
            }
        }
        if (std::chrono::steady_clock::now() - last > std::chrono::seconds(20)) {
            report(hist, total, "");
            if (rtotal) report(rhist, rtotal, "-raster");
            hist.clear(); total = 0; rhist.clear(); rtotal = 0;
            last = std::chrono::steady_clock::now();
        }
    }
}

void profiler_start() {
    g_prof_on = true;
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &g_target,
                    THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0);
    std::thread(sampler).detach();
}
