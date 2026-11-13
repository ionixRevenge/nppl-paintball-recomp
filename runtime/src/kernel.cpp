// High-level EE kernel: syscalls, threads (fibers), semaphores, interrupt handlers.
//
// The PS2 kernel schedules threads cooperatively by priority (lower number =
// higher priority); a thread runs until it blocks or yields. Each guest thread
// gets its own host fiber and R5900 context. Interrupts (vblank, DMA) are
// delivered at "safe points": syscalls and polled hardware reads.
#include <xmmintrin.h>
#include "rt_internal.h"
#include "hw.h"
#include "sif.h"
#include <windows.h>
#include <timeapi.h>
#include <chrono>
#include <deque>
#include <string>
#include <map>
#include <vector>

enum ThreadState { TH_FREE, TH_DORMANT, TH_READY, TH_RUNNING, TH_WAIT, TH_SUSPEND, TH_WAIT_SUSPEND };
enum WaitType { W_NONE, W_SLEEP, W_SEMA };

struct Thread {
    ThreadState state = TH_FREE;
    WaitType wait = W_NONE;
    int wait_id = 0;
    uint32_t entry = 0, stack = 0, stack_size = 0, gp = 0, attr = 0, option = 0;
    int init_prio = 0, prio = 0;
    int wakeup_count = 0;
    uint32_t arg = 0;
    R5900 *ctx = nullptr;
    void *fiber = nullptr;
};

struct Sema { bool used = false; int count = 0, max = 0, init = 0; uint32_t attr = 0, option = 0; std::deque<int> waiters; };
struct Handler { bool used = false; int cause = 0; uint32_t func = 0, arg = 0; bool dmac = false; };

static const int MAX_THREADS = 256, MAX_SEMAS = 256, MAX_HANDLERS = 64;
static Thread g_th[MAX_THREADS];
static Sema g_sema[MAX_SEMAS];
static Handler g_handlers[MAX_HANDLERS];
static uint32_t g_intc_enabled, g_dmac_enabled;
static int g_cur = 1;              // current thread id
static void *g_sched_fiber;        // fiber of the host main thread (scheduler)
static bool g_in_interrupt;
static int g_irq_disable;
static R5900 g_irq_ctx;
static const uint32_t IRQ_STACK_TOP = 0x00080000;  // kernel RAM, unused by the game
static int64_t g_last_vsync;   // emulated ns (rt_emu_ns)
static uint32_t g_heap_end;

static int ready_best();
static uint64_t g_irq_count[16];
static void reschedule();

// ------------------------------------------------------------ interrupts
static void call_guest(uint32_t func, uint32_t a0, uint32_t a1, uint32_t a2) {
    R5900 *c = &g_irq_ctx;
    memset(c->r, 0, sizeof c->r);
    c->r[4].ud[0] = SX32(a0); c->r[5].ud[0] = SX32(a1); c->r[6].ud[0] = SX32(a2);
    c->r[28] = g_th[g_cur].ctx ? g_th[g_cur].ctx->r[28] : c->r[28];
    c->r[29].ud[0] = IRQ_STACK_TOP - 0x100;
    c->r[31].ud[0] = 0;
    rt_call(c, func);
}

// An interrupt raised while interrupts are disabled (DI) or while another handler runs stays
// pending, as the INTC/DMAC STAT bit does on hardware, and is taken at the next safepoint
// with interrupts enabled. Dropping it instead lost timer ticks: the 30 Hz engine tick
// handler (TIM1 compare) resets COUNT, so one lost compare stalled the tick for 65536
// hblanks (~4 s) until the counter wrapped round to COMP again.
static uint32_t g_intc_pending, g_dmac_pending;
uint64_t g_irq_deferred[16];   // stats: interrupts that had to wait for EI / handler exit

static void fire_intc(int cause) {
    hw_raise_intc(cause);
    if (!(g_intc_enabled & (1u << cause))) return;
    if (g_in_interrupt || g_irq_disable) { g_intc_pending |= 1u << cause; g_irq_deferred[cause & 15]++; return; }
    g_in_interrupt = true;
    g_irq_count[cause & 15]++;
    for (auto &h : g_handlers)
        if (h.used && !h.dmac && h.cause == cause) call_guest(h.func, cause, h.arg, 0);
    g_in_interrupt = false;
}

void kernel_intc_event(int cause) { fire_intc(cause); }

void kernel_dmac_event(int ch) {
    if (!(g_dmac_enabled & (1u << ch))) return;
    if (g_in_interrupt || g_irq_disable) { g_dmac_pending |= 1u << ch; return; }
    g_in_interrupt = true;
    for (auto &h : g_handlers)
        if (h.used && h.dmac && h.cause == ch) call_guest(h.func, ch, h.arg, 0);
    g_in_interrupt = false;
}

static bool deliver_pending() {   // caller checked !g_in_interrupt && !g_irq_disable
    if (!g_intc_pending && !g_dmac_pending) return false;
    uint32_t ip = g_intc_pending, dp = g_dmac_pending;
    g_intc_pending = g_dmac_pending = 0;
    for (int i = 0; i < 16 && !g_irq_disable; i++)
        if (ip & (1u << i)) { ip &= ~(1u << i); fire_intc(i); }
    for (int i = 0; i < 16 && !g_irq_disable; i++)
        if (dp & (1u << i)) { dp &= ~(1u << i); kernel_dmac_event(i); }
    g_intc_pending |= ip; g_dmac_pending |= dp;   // a handler left interrupts disabled
    return true;
}

static void maybe_preempt() {
    // an interrupt handler (or SIF reply) may have woken a higher-priority thread
    int best = ready_best();
    if (best && g_th[best].prio < g_th[g_cur].prio && g_th[g_cur].state == TH_RUNNING) {
        g_th[g_cur].state = TH_READY;
        reschedule();
    }
}

// Deliver pending SIF replies, timer interrupts and vblank (59.94 Hz of emulated time).
// Vblanks stay phase-locked to the clock: a late one does not push the next one back (that
// made the game see ~31 vblanks/s at ~31 fps), and after a long stall the schedule resyncs
// instead of firing a burst.
static const int64_t VSYNC_NS = 16683350;
static bool g_vsync_old = getenv("NPPL_VSYNC_OLD") && atoi(getenv("NPPL_VSYNC_OLD")) != 0;  // debug: old timing
void kernel_safepoint() {
    if (g_in_interrupt) return;
    bool any = false;
    if (!g_irq_disable) { any |= deliver_pending(); any |= sif_deliver_one(); }
    if (uint32_t t = hw_timer_poll()) {
        any = true;
        for (int line = INTC_TIM0; line <= INTC_TIM3; line++)
            if (t & (1u << line)) fire_intc(line);
    }
    int64_t now = (int64_t)rt_emu_ns();
    if (now - g_last_vsync >= VSYNC_NS) {
        if (g_vsync_old) g_last_vsync = now;
        else {
            g_last_vsync += VSYNC_NS;
            if (now - g_last_vsync >= 2 * VSYNC_NS) g_last_vsync = now - (now - g_last_vsync) % VSYNC_NS;
        }
        path3_idle();
        fire_intc(INTC_VBLANK_START);
        hw_vsync_tick();
        fire_intc(INTC_VBLANK_END);
        any = true;
    }
    if (any && !g_irq_disable) maybe_preempt();
}

// ------------------------------------------------------------ scheduler
static int ready_best() {
    int best = 0;
    for (int i = 1; i < MAX_THREADS; i++)
        if (g_th[i].state == TH_READY && (!best || g_th[i].prio < g_th[best].prio)) best = i;
    return best;
}

static void wait_for_ready() {
    // Nothing runnable: let time pass so vblank handlers can wake someone.
    for (;;) {
        int best = ready_best();
        if (best) return;
        if (sif_has_pending()) { kernel_safepoint(); continue; }
        if (g_vsync_old) g_last_vsync -= 17000000;  // old behaviour: force the next vblank now
        kernel_safepoint();
        if (!ready_best()) {
            static int idle_spins;
            if (++idle_spins % 600 == 0) rt_log(LOG_WARN, "all threads blocked (deadlock?)");
            Sleep(1);
        }
    }
}

// Switch away from the current thread (which must no longer be RUNNING).
static void reschedule() {
    wait_for_ready();
    int next = ready_best();
    // round-robin among equal priorities: prefer a thread other than current
    if (g_th[g_cur].state == TH_READY && g_th[next].prio == g_th[g_cur].prio) {
        for (int i = 1; i < MAX_THREADS; i++) {
            int k = (g_cur + i) % MAX_THREADS;
            if (k && g_th[k].state == TH_READY && g_th[k].prio == g_th[next].prio) { next = k; break; }
        }
    }
    g_th[next].state = TH_RUNNING;
    if (next == g_cur) return;
    int prev = g_cur;
    g_cur = next;
    (void)prev;
    SwitchToFiber(g_th[next].fiber);
}

static void make_ready(int id) {
    Thread &t = g_th[id];
    if (t.state == TH_WAIT) t.state = TH_READY;
    else if (t.state == TH_WAIT_SUSPEND) t.state = TH_SUSPEND;
    t.wait = W_NONE;
}

static void block_current(WaitType w, int id) {
    Thread &t = g_th[g_cur];
    t.state = TH_WAIT; t.wait = w; t.wait_id = id;
    reschedule();
}

static void __stdcall thread_fiber_main(void *param) {
    int id = (int)(intptr_t)param;
    Thread &t = g_th[id];
    R5900 *c = t.ctx;
    rt_log(LOG_INFO, "thread %d starting at %08X (prio %d)", id, t.entry, t.prio);
    rt_call(c, t.entry);
    // returned from thread entry: ExitThread
    rt_log(LOG_INFO, "thread %d exited", id);
    t.state = TH_DORMANT;
    reschedule();
    rt_log(LOG_ERR, "dormant thread resumed"); rt_exit(1);
}

static void thread_start(int id, uint32_t arg) {
    Thread &t = g_th[id];
    if (!t.ctx) t.ctx = new R5900();
    memset(t.ctx, 0, sizeof(R5900));
    R5900 *c = t.ctx;
    c->thread_id = id;
    c->r[4].ud[0] = SX32(arg);
    c->r[28].ud[0] = SX32(t.gp);
    c->r[29].ud[0] = SX32(t.stack + t.stack_size - 0x10);
    c->vf[0] = VF0;
    t.prio = t.init_prio;
    t.wakeup_count = 0;
    if (t.fiber) DeleteFiber(t.fiber);
    t.fiber = CreateFiber(1 << 20, thread_fiber_main, (void *)(intptr_t)id);
    t.state = TH_READY;
}

// ------------------------------------------------------------ syscalls
static uint32_t A(R5900 *c, int n) { return (uint32_t)c->r[4 + n].ud[0]; }
static void RET(R5900 *c, int64_t v) { c->r[2].ud[0] = (uint64_t)v; }

static const char *syscall_name(int n) {
    switch (n) {
    case 0x02: return "SetGsCrt"; case 0x04: return "Exit"; case 0x06: return "LoadExecPS2";
    case 0x10: return "AddIntcHandler"; case 0x11: return "RemoveIntcHandler";
    case 0x12: return "AddDmacHandler"; case 0x13: return "RemoveDmacHandler";
    case 0x14: return "_EnableIntc"; case 0x15: return "_DisableIntc";
    case 0x16: return "_EnableDmac"; case 0x17: return "_DisableDmac";
    case 0x20: return "CreateThread"; case 0x21: return "DeleteThread"; case 0x22: return "StartThread";
    case 0x23: return "ExitThread"; case 0x24: return "ExitDeleteThread"; case 0x25: return "TerminateThread";
    case 0x29: return "ChangeThreadPriority"; case 0x2B: return "RotateThreadReadyQueue";
    case 0x2F: return "GetThreadId"; case 0x30: return "ReferThreadStatus";
    case 0x32: return "SleepThread"; case 0x33: return "WakeupThread"; case 0x34: return "iWakeupThread";
    case 0x37: return "SuspendThread"; case 0x39: return "ResumeThread";
    case 0x3C: return "SetupThread"; case 0x3D: return "SetupHeap"; case 0x3E: return "EndOfHeap";
    case 0x40: return "CreateSema"; case 0x41: return "DeleteSema"; case 0x42: return "SignalSema";
    case 0x43: return "iSignalSema"; case 0x44: return "WaitSema"; case 0x45: return "PollSema";
    case 0x46: return "iPollSema"; case 0x47: return "ReferSemaStatus";
    case 0x64: return "FlushCache"; case 0x70: return "GsGetIMR"; case 0x71: return "GsPutIMR";
    case 0x73: return "SetVSyncFlag"; case 0x74: return "SetSyscall"; case 0x76: return "SifDmaStat";
    case 0x77: return "SifSetDma"; case 0x78: return "SifSetDChain"; case 0x79: return "SifSetReg";
    case 0x7A: return "SifGetReg"; case 0x7C: return "Deci2Call"; case 0x7E: return "MachineType";
    case 0x7F: return "GetMemorySize"; case 0x5A: return "Copy"; case 0x5B: return "GetEntryAddress";
    case 0x83: return "FindAddress"; case 0x4A: return "SetOsdConfigParam"; case 0x6B: return "SifStopDma"; case 0x4B: return "GetOsdConfigParam"; case 0x6E: return "SetOsdConfigParam2"; case 0x6F: return "GetOsdConfigParam2";
    }
    return "?";
}

uint32_t sif_set_dma(uint32_t desc, uint32_t count);  // sif.cpp
uint32_t sif_get_reg(uint32_t reg);
void sif_set_reg(uint32_t reg, uint32_t v);

static uint32_t g_gs_imr = 0x7F00;

// Syscall table in guest kernel memory: the SDK locates it by searching for
// the addresses of handlers it installed with SetSyscall.
static const uint32_t SYSCALL_TABLE = 0x80010000;
static std::map<int, uint32_t> g_user_sys;
// OSD config: 4:3, RGB, English (language field = 1 at bit 16), version 1
static uint32_t g_osd_config = (1u << 16) | (1u << 13);
extern "C" bool rt_has_func(uint32_t addr);
static bool is_native(int n) {
    switch (n) {
    case 0x02: case 0x04: case 0x06: case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15:
    case 0x16: case 0x17: case 0x20: case 0x21: case 0x22: case 0x23: case 0x24: case 0x25: case 0x29:
    case 0x2B: case 0x2F: case 0x30: case 0x32: case 0x33: case 0x34: case 0x37: case 0x39: case 0x3C:
    case 0x3D: case 0x3E: case 0x40: case 0x41: case 0x42: case 0x43: case 0x44: case 0x45: case 0x46:
    case 0x47: case 0x64: case 0x70: case 0x71: case 0x73: case 0x74: case 0x76: case 0x77: case 0x78:
    case 0x79: case 0x7A: case 0x7C: case 0x7E: case 0x7F: case 0x6E: case 0x6F:
        return true;
    }
    return false;
}

extern "C" void rt_syscall(R5900 *c, uint32_t pc) {
    int n = (int)(int32_t)c->r[3].ud[0];
    if (n < 0) n = -n;   // i* (interrupt-context) variants
    if (rt_once(0x400000000ull | (uint32_t)n))
        rt_log(LOG_INFO, "syscall %02X %s (first call from %08X)", n, syscall_name(n), pc);
    rt_log(LOG_TRACE, "syscall %02X %s(%08X, %08X, %08X, %08X) @%08X", n, syscall_name(n), A(c, 0), A(c, 1), A(c, 2), A(c, 3), pc);
    switch (n) {
    case 0x02: rt_log(LOG_INFO, "SetGsCrt(interlace=%d, mode=%d, field=%d)", A(c, 0), A(c, 1), A(c, 2)); RET(c, 0); break;
    case 0x04: rt_log(LOG_INFO, "Exit(%d)", A(c, 0)); rt_exit((int)A(c, 0));
    case 0x06: rt_log(LOG_ERR, "LoadExecPS2 not supported"); rt_exit(1);
    case 0x10: case 0x12: {
        for (int i = 1; i < MAX_HANDLERS; i++)
            if (!g_handlers[i].used) {
                g_handlers[i] = Handler{true, (int)A(c, 0), A(c, 1), A(c, 3), n == 0x12};
                RET(c, i); return;
            }
        RET(c, -1); break;
    }
    case 0x11: case 0x13: if (A(c, 1) < MAX_HANDLERS) g_handlers[A(c, 1)].used = false; RET(c, 0); break;
    case 0x14: g_intc_enabled |= 1u << A(c, 0); RET(c, 1); break;
    case 0x15: g_intc_enabled &= ~(1u << A(c, 0)); RET(c, 1); break;
    case 0x16: g_dmac_enabled |= 1u << A(c, 0); RET(c, 1); break;
    case 0x17: g_dmac_enabled &= ~(1u << A(c, 0)); RET(c, 1); break;

    case 0x20: {  // CreateThread(ee_thread_t*)
        uint32_t p = A(c, 0);
        for (int i = 2; i < MAX_THREADS; i++)
            if (g_th[i].state == TH_FREE) {
                Thread &t = g_th[i];
                t = Thread{};
                t.state = TH_DORMANT;
                t.entry = LW(p + 4); t.stack = LW(p + 8); t.stack_size = LW(p + 12);
                t.gp = LW(p + 16); t.init_prio = t.prio = (int)LW(p + 20);
                t.attr = LW(p + 28); t.option = LW(p + 32);
                rt_log(LOG_INFO, "CreateThread -> %d entry=%08X stack=%08X+%X prio=%d", i, t.entry, t.stack, t.stack_size, t.prio);
                RET(c, i); return;
            }
        RET(c, -1); break;
    }
    case 0x21: if (A(c, 0) < MAX_THREADS) g_th[A(c, 0)].state = TH_FREE; RET(c, 0); break;
    case 0x22: {
        int id = (int)A(c, 0);
        thread_start(id, A(c, 1));
        RET(c, 0);
        if (g_th[id].prio < g_th[g_cur].prio) { g_th[g_cur].state = TH_READY; reschedule(); }
        break;
    }
    case 0x23: case 0x24:
        g_th[g_cur].state = n == 0x24 ? TH_FREE : TH_DORMANT;
        reschedule();
        rt_log(LOG_ERR, "exited thread resumed"); rt_exit(1);
    case 0x25: {
        int id = (int)A(c, 0);
        if (id > 0 && id < MAX_THREADS) g_th[id].state = TH_DORMANT;
        RET(c, 0); break;
    }
    case 0x29: {
        int id = (int)A(c, 0); if (id == 0) id = g_cur;
        int old = g_th[id].prio; g_th[id].prio = (int)A(c, 1);
        RET(c, old);
        int best = ready_best();
        if (best && g_th[best].prio < g_th[g_cur].prio) { g_th[g_cur].state = TH_READY; reschedule(); }
        break;
    }
    case 0x2B:  // RotateThreadReadyQueue(prio): yield to same-priority threads
        RET(c, 0);
        g_th[g_cur].state = TH_READY; reschedule();
        break;
    case 0x2F: RET(c, g_cur); break;
    case 0x30: {  // ReferThreadStatus(id, ee_thread_status_t*)
        int id = (int)A(c, 0); if (id == 0) id = g_cur;
        uint32_t p = A(c, 1);
        if (id <= 0 || id >= MAX_THREADS || g_th[id].state == TH_FREE) { RET(c, -1); break; }
        Thread &t = g_th[id];
        static const int st[] = {0, 0x10, 0x02, 0x01, 0x04, 0x08, 0x0C};
        if (p) {
            SW(p + 0, st[t.state]); SW(p + 4, t.entry); SW(p + 8, t.stack); SW(p + 12, t.stack_size);
            SW(p + 16, t.gp); SW(p + 20, t.init_prio); SW(p + 24, t.prio); SW(p + 28, t.attr); SW(p + 32, t.option);
            SW(p + 36, t.wait); SW(p + 40, t.wait_id); SW(p + 44, t.wakeup_count);
        }
        RET(c, st[t.state]); break;
    }
    case 0x32:  // SleepThread
        RET(c, g_cur);
        if (g_th[g_cur].wakeup_count > 0) { g_th[g_cur].wakeup_count--; break; }
        block_current(W_SLEEP, 0);
        break;
    case 0x33: case 0x34: {
        int id = (int)A(c, 0);
        Thread &t = g_th[id];
        if (t.state == TH_WAIT && t.wait == W_SLEEP) make_ready(id); else t.wakeup_count++;
        RET(c, id);
        if (n == 0x33 && t.state == TH_READY && t.prio < g_th[g_cur].prio) { g_th[g_cur].state = TH_READY; reschedule(); }
        break;
    }
    case 0x37: { int id = (int)A(c, 0); Thread &t = g_th[id];
        if (t.state == TH_READY) t.state = TH_SUSPEND; else if (t.state == TH_WAIT) t.state = TH_WAIT_SUSPEND;
        RET(c, id); break; }
    case 0x39: { int id = (int)A(c, 0); Thread &t = g_th[id];
        if (t.state == TH_SUSPEND) t.state = TH_READY; else if (t.state == TH_WAIT_SUSPEND) t.state = TH_WAIT;
        RET(c, id); break; }

    case 0x3C: {  // SetupThread(gp, stack, stack_size, args, root)
        uint32_t stack = A(c, 1), size = A(c, 2);
        uint32_t top = (stack == 0xFFFFFFFFu) ? RAM_SIZE : stack + size;
        Thread &t = g_th[1];
        t.stack = top - size; t.stack_size = size; t.gp = A(c, 0);
        rt_log(LOG_INFO, "SetupThread: gp=%08X stack=%08X size=%X -> sp=%08X", A(c, 0), stack, size, top - 0x2A0);
        RET(c, top - 0x2A0); break;
    }
    case 0x3D: {  // SetupHeap(start, size)
        uint32_t start = A(c, 0), size = A(c, 1);
        g_heap_end = (size == 0xFFFFFFFFu) ? g_th[1].stack : start + size;
        rt_log(LOG_INFO, "SetupHeap: %08X..%08X", start, g_heap_end);
        RET(c, g_heap_end); break;
    }
    case 0x3E: RET(c, g_heap_end); break;

    case 0x40: {  // CreateSema(ee_sema_t*)
        uint32_t p = A(c, 0);
        for (int i = 1; i < MAX_SEMAS; i++)
            if (!g_sema[i].used) {
                Sema &s = g_sema[i];
                s = Sema{};
                s.used = true; s.count = (int)LW(p + 0); s.max = (int)LW(p + 4); s.init = (int)LW(p + 8);
                s.attr = LW(p + 16); s.option = LW(p + 20);
                s.count = s.init;
                RET(c, i); return;
            }
        RET(c, -1); break;
    }
    case 0x41: if (A(c, 0) < MAX_SEMAS) {
            Sema &s = g_sema[A(c, 0)];
            for (int w : s.waiters) make_ready(w);
            s = Sema{};
        }
        RET(c, A(c, 0)); break;
    case 0x42: case 0x43: {  // SignalSema
        int id = (int)A(c, 0);
        if (id <= 0 || id >= MAX_SEMAS || !g_sema[id].used) { RET(c, -1); break; }
        Sema &s = g_sema[id];
        RET(c, id);
        if (!s.waiters.empty()) {
            int w = s.waiters.front(); s.waiters.pop_front();
            make_ready(w);
            if (n == 0x42 && g_th[w].state == TH_READY && g_th[w].prio < g_th[g_cur].prio) {
                g_th[g_cur].state = TH_READY; reschedule();
            }
        } else if (s.max <= 0 || s.count < s.max) s.count++;   // capped at max_count (overflow is dropped)
        break;
    }
    case 0x44: {  // WaitSema
        int id = (int)A(c, 0);
        if (id <= 0 || id >= MAX_SEMAS || !g_sema[id].used) { RET(c, -1); break; }
        Sema &s = g_sema[id];
        RET(c, id);
        if (s.count > 0) { s.count--; break; }
        s.waiters.push_back(g_cur);
        block_current(W_SEMA, id);
        break;
    }
    case 0x45: case 0x46: {  // PollSema
        int id = (int)A(c, 0);
        if (id <= 0 || id >= MAX_SEMAS || !g_sema[id].used) { RET(c, -1); break; }
        Sema &s = g_sema[id];
        if (s.count > 0) { s.count--; RET(c, id); } else RET(c, -1);
        break;
    }
    case 0x47: {
        int id = (int)A(c, 0); uint32_t p = A(c, 1);
        if (id <= 0 || id >= MAX_SEMAS || !g_sema[id].used) { RET(c, -1); break; }
        Sema &s = g_sema[id];
        if (p) { SW(p, s.count); SW(p + 4, s.max); SW(p + 8, s.init); SW(p + 12, (uint32_t)s.waiters.size()); SW(p + 16, s.attr); SW(p + 20, s.option); }
        RET(c, id); break;
    }

    case 0x4A: g_osd_config = LW(A(c, 0)); RET(c, 0); break;  // SetOsdConfigParam
    case 0x4B: SW(A(c, 0), g_osd_config); RET(c, 0); break;   // GetOsdConfigParam
    case 0x6E: RET(c, 0); break;                    // SetOsdConfigParam2: ignored
    case 0x6F: {                                    // GetOsdConfigParam2(buf, size, offset): defaults (zeros)
        uint32_t buf = A(c, 0), size = A(c, 1);
        for (uint32_t k = 0; k < size && k < 128; k++) SB(buf + k, 0);
        RET(c, 0); break;
    }
    case 0x64: RET(c, 0); break;                    // FlushCache
    case 0x70: RET(c, g_gs_imr); break;             // GsGetIMR
    case 0x71: { uint32_t o = g_gs_imr; g_gs_imr = A(c, 0); RET(c, o); break; }
    case 0x73: RET(c, 0); break;                    // SetVSyncFlag
    case 0x74: {  // SetSyscall(num, handler): game-installed syscall handler
        uint32_t num = A(c, 0) & 0xFF, fn = A(c, 1);
        g_user_sys[num] = fn;
        SW(SYSCALL_TABLE + num * 4, fn);
        rt_log(LOG_INFO, "SetSyscall(%02X, %08X)%s", num, fn, is_native(num) ? " (native HLE kept)" : "");
        RET(c, 0); break;
    }
    case 0x76: RET(c, -1); break;                   // SifDmaStat: completed
    case 0x77: RET(c, sif_set_dma(A(c, 0), A(c, 1))); break;
    case 0x78: RET(c, 0); break;                    // SifSetDChain
    case 0x79: sif_set_reg(A(c, 0), A(c, 1)); RET(c, 0); break;
    case 0x7A: RET(c, (int32_t)sif_get_reg(A(c, 0))); break;
    case 0x7C: {  // Deci2Call(func, param): debug channel; 0x10 = kputs
        uint32_t fn = A(c, 0), prm = A(c, 1);
        if (rt_once(0x520000000ull | fn)) rt_log(LOG_INFO, "Deci2Call func %X", fn);
        if (fn == 0x10 && prm) {
            uint32_t sp = LW(prm);
            std::string str;
            for (uint32_t k = 0; k < 4096; k++) { char ch = (char)LB(sp + k); if (!ch) break; str += ch; }
            while (!str.empty() && (str.back() == 10 || str.back() == 13)) str.pop_back();
            rt_log(LOG_INFO, "TTY: %s", str.c_str());
        }
        RET(c, -1); break;  // no debugger attached (retail): every request fails
    }
    case 0x6B: RET(c, 0); break;                    // SifStopDma
    case 0x0D: case 0x0E: RET(c, 0); break;         // SetVTLBRefillHandler / SetVCommonHandler
    case 0x7E: RET(c, 0); break;                    // MachineType: retail
    case 0x7F: RET(c, RAM_SIZE); break;             // GetMemorySize
    default:
        if (g_user_sys.count(n) && g_user_sys[n] && !rt_has_func(g_user_sys[n])) {
            // kernel-mode patch code copied in at runtime (not recompiled):
            // the HLE kernel already provides what it patches
            if (rt_once(0x510000000ull | (uint32_t)n))
                rt_log(LOG_INFO, "syscall %02X: skipping kernel patch at %08X", n, g_user_sys[n]);
            RET(c, 0); break;
        }
        if (g_user_sys.count(n) && g_user_sys[n]) {
            // run the game's own handler, like the kernel would
            uint64_t ra = c->r[31].ud[0];
            rt_call(c, g_user_sys[n]);
            c->r[31].ud[0] = ra;
            break;
        }
        if (rt_once(0x500000000ull | (uint32_t)n)) rt_log(LOG_WARN, "unimplemented syscall %02X at %08X", n, pc);
        RET(c, 0);
    }
    kernel_safepoint();
}

// rate report: interrupts taken (and deferred past a DI / handler) per cause since the last call
std::string kernel_irq_rates(double el) {
    static uint64_t last[16], lastd[16];
    char b[160];
    snprintf(b, sizeof b, "vbl %.1f tim0 %.1f tim1 %.1f /s (deferred vbl %llu tim1 %llu)",
             (g_irq_count[2] - last[2]) / el, (g_irq_count[9] - last[9]) / el, (g_irq_count[10] - last[10]) / el,
             (unsigned long long)(g_irq_deferred[2] - lastd[2]), (unsigned long long)(g_irq_deferred[10] - lastd[10]));
    memcpy(last, g_irq_count, sizeof last); memcpy(lastd, g_irq_deferred, sizeof lastd);
    return b;
}

std::string kernel_debug_state() {
    char b[256];
    snprintf(b, sizeof b, "irqdis=%d inirq=%d intc_en=%X vbl=%llu vif1=%llu tim1=%llu", g_irq_disable, (int)g_in_interrupt,
             g_intc_enabled, (unsigned long long)g_irq_count[2], (unsigned long long)g_irq_count[5], (unsigned long long)g_irq_count[10]);
    std::string out = b;
    for (int i = 1; i < MAX_THREADS; i++)
        if (g_th[i].state != TH_FREE) {
            snprintf(b, sizeof b, " t%d:s%d/p%d", i, (int)g_th[i].state, g_th[i].prio);
            out += b;
        }
    return out;
}

extern "C" void rt_ei(R5900 *) { if (g_irq_disable > 0) g_irq_disable = 0; }
extern "C" void rt_di(R5900 *) { g_irq_disable = 1; }
extern "C" void rt_eret(R5900 *) {}

void kernel_init(R5900 *main_ctx) {
    // The EE FPU (and the VUs) truncate results (round toward zero) and treat denormals as zero.
    // All PS2 code runs as fibers on this host thread, so the SSE control register could match
    // that (round toward zero, flush-to-zero, denormals-are-zero). Off by default: it made no
    // difference to the stuck-character bug. NPPL_FPUROUND=1 enables it.
    if (getenv("NPPL_FPUROUND") && atoi(getenv("NPPL_FPUROUND")) != 0)
        _mm_setcsr((_mm_getcsr() & ~0x6000u) | 0x6000u | 0x8040u);
    timeBeginPeriod(1);   // 1 ms Sleep() while idle-waiting for the next vblank
    g_sched_fiber = ConvertThreadToFiber(nullptr);
    Thread &t = g_th[1];
    t.state = TH_RUNNING; t.prio = t.init_prio = 0; t.ctx = main_ctx; t.fiber = g_sched_fiber;
    main_ctx->thread_id = 1;
    g_cur = 1;
    memset(&g_irq_ctx, 0, sizeof g_irq_ctx);
    g_irq_ctx.vf[0] = VF0;
}

int kernel_current_thread() { return g_cur; }
R5900 *kernel_current_ctx() { return g_th[g_cur].ctx; }

uint32_t rt_loop_counter;
extern "C" void rt_loop_hook(R5900 *c, uint32_t target) {
    rt_loop_counter = 0;
    c->pc = target;
    kernel_safepoint();
}
