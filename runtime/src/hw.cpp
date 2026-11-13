// EE hardware registers (0x10000000-0x1000FFFF) and GS privileged registers (0x12000000).
// Devices are modelled at the level needed to keep the SCE libraries happy;
// DMA transfers complete instantly and are forwarded to device handlers.
#include "rt_internal.h"
#include "hw.h"
#include <chrono>
#include <algorithm>

static uint32_t g_ee_regs[0x10000 / 4];
static uint64_t g_gs_priv[0x2000 / 8];
uint32_t gs_csr_events();   // gs.cpp: SIGNAL/FINISH bits
static auto g_t0 = std::chrono::steady_clock::now();

static uint64_t bus_cycles() {  // 147.456 MHz bus clock
    return (uint64_t)((double)rt_emu_ns() * 0.147456);
}

static uint32_t &reg(uint32_t p) { return g_ee_regs[(p & 0xFFFF) >> 2]; }

// ---- EE timers T0..T3 (COUNT +0, MODE +0x10, COMP +0x20, HOLD +0x30)
// MODE: CLKS 0-1 (bus, bus/16, bus/256, hblank), ZRET 6, CUE 7, CMPE 8, OVFE 9, EQUF 10, OVFF 11
// late: ticks by which the last compare interrupt was delivered after the match. Interrupts
// only reach the game at safepoints, so a handler that restarts the counter (COUNT = 0, as the
// game's 30 Hz engine tick on T1 does) would otherwise lose that time on every tick and the
// tick would run slow. The next COUNT write carries it over (bounded to a few periods).
struct Timer { uint32_t mode, comp; uint64_t base_ticks, base_cyc, last_ovf, last_cmp, late; };
static Timer g_tim[4];
static const uint32_t TIM_BASE[4] = {0x10000000, 0x10000800, 0x10001000, 0x10001800};
static int timer_index(uint32_t p) {
    for (int i = 0; i < 4; i++) if ((p & ~0x3Fu) == TIM_BASE[i]) return i;
    return -1;
}
static uint64_t timer_div(const Timer &t) {
    static const uint64_t d[4] = {1, 16, 256, 9372};  // 147.456 MHz / 15734 Hz hblank
    return d[t.mode & 3];
}
static uint64_t timer_ticks(const Timer &t) {   // ticks since the counter was last written
    if (!(t.mode & 0x80)) return t.base_ticks;
    return t.base_ticks + (bus_cycles() - t.base_cyc) / timer_div(t);
}
static uint32_t timer_period(const Timer &t) {   // ZRET: counter resets on reaching COMP
    return (t.mode & 0x40) && t.comp ? t.comp + 1 : 0x10000;
}
static uint32_t timer_count(const Timer &t) { return (uint32_t)(timer_ticks(t) % timer_period(t)); }
static uint64_t timer_cmp_index(const Timer &t, uint64_t ticks) {   // compare matches in [0, ticks]
    uint64_t per = timer_period(t);
    return (ticks + per - (t.comp % per)) / per;
}
// carry: ticks to add on top of `ticks` without counting them as already-seen events
static void timer_rebase(Timer &t, uint64_t ticks, uint64_t carry = 0) {
    t.base_ticks = ticks + carry; t.base_cyc = bus_cycles();
    t.last_ovf = ticks / 0x10000; t.last_cmp = timer_cmp_index(t, ticks);
}
// Called at kernel safepoints: returns the INTC lines (9..12) that fired.
uint32_t hw_timer_poll() {
    uint32_t lines = 0;
    for (int i = 0; i < 4; i++) {
        Timer &t = g_tim[i];
        if (!(t.mode & 0x80) || !(t.mode & 0x300)) continue;
        uint64_t ticks = timer_ticks(t);
        if (t.mode & 0x100) {   // compare
            uint64_t per = timer_period(t), n = timer_cmp_index(t, ticks);
            if (n != t.last_cmp) {
                uint64_t match = (n - 1) * per + t.comp % per;   // tick of the latest match
                t.late = ticks > match ? ticks - match : 0;
                t.last_cmp = n; t.mode |= 0x400; lines |= 1u << (9 + i);
            }
        }
        if ((t.mode & 0x200) && !(t.mode & 0x40)) {   // overflow
            uint64_t n = ticks / 0x10000;
            if (n != t.last_ovf) { t.last_ovf = n; t.mode |= 0x800; lines |= 1u << (9 + i); }
        }
    }
    return lines;
}
static uint32_t timer_read(int i, uint32_t off) {
    Timer &t = g_tim[i];
    switch (off) {
    case 0x00:
        // The game's clock is (overflow interrupts x 0xFFFF) + COUNT (T0, OVFE). Until the
        // overflow interrupt has run, a wrapped COUNT would make that clock jump back ~4 s;
        // take the interrupt now if possible, else hold COUNT at its maximum.
        if ((t.mode & 0x280) == 0x280 && !(t.mode & 0x40) && timer_ticks(t) / 0x10000 != t.last_ovf) {
            kernel_safepoint();
            if (timer_ticks(t) / 0x10000 != t.last_ovf) return 0xFFFF;
        }
        return timer_count(t);
    case 0x10: return t.mode;
    case 0x20: return t.comp;
    default: return 0;
    }
}
static void timer_write(int i, uint32_t off, uint32_t v) {
    Timer &t = g_tim[i];
    switch (off) {
    case 0x00: {
        uint64_t carry = 0;
        if (t.late && (t.mode & 0x100)) carry = std::min<uint64_t>(t.late, 4ull * (t.comp ? t.comp : 1));
        t.late = 0;
        timer_rebase(t, v & 0xFFFF, carry);
        break;
    }
    case 0x10: {
        uint64_t cur = timer_ticks(t) % timer_period(t);
        t.mode = (t.mode & 0xC00 & ~(v & 0xC00)) | (v & 0x3FF);   // EQUF/OVFF: write 1 to clear
        timer_rebase(t, cur);
        break;
    }
    case 0x20: t.comp = v & 0xFFFF; timer_rebase(t, timer_ticks(t) % 0x10000); break;
    }
}

void hw_init() {
    memset(g_ee_regs, 0, sizeof g_ee_regs);
    // SIF: IOP side reports "booted, SIF initialised"
    reg(SIF_SMFLG) = 0x00010000u | 0x00020000u | 0x00040000u;
    // Timer 3 "running": tells libkernel the kernel already provides alarms,
    // so it skips installing its kernel-mode alarm patch (alarms are HLE'd).
    reg(0x10001810) = 0x100;
    g_gs_priv[(GS_CSR & 0x1FFF) >> 3] = 0x551B0000ull;  // GS revision/id
}

static const char *dma_name(int ch) {
    static const char *n[] = {"VIF0", "VIF1", "GIF", "fromIPU", "toIPU", "SIF0", "SIF1", "SIF2", "fromSPR", "toSPR"};
    return ch >= 0 && ch < 10 ? n[ch] : "?";
}

static int dma_channel(uint32_t p) {
    switch (p & 0xFFFFFF00u) {
    case 0x10008000: return 0; case 0x10009000: return 1; case 0x1000A000: return 2;
    case 0x1000B000: return 3; case 0x1000B400: return 4; case 0x1000C000: return 5;
    case 0x1000C400: return 6; case 0x1000C800: return 7; case 0x1000D000: return 8;
    case 0x1000D400: return 9;
    }
    return -1;
}

static uint32_t dma_base(int ch) {
    static const uint32_t b[] = {0x10008000, 0x10009000, 0x1000A000, 0x1000B000, 0x1000B400,
                                 0x1000C000, 0x1000C400, 0x1000C800, 0x1000D000, 0x1000D400};
    return b[ch];
}

static void dma_start(int ch) {
    uint32_t base = dma_base(ch);
    uint32_t chcr = reg(base), madr = reg(base + 0x10), qwc = reg(base + 0x20), tadr = reg(base + 0x30);
    rt_log(LOG_TRACE, "DMA %s start chcr=%08X madr=%08X qwc=%X tadr=%08X", dma_name(ch), chcr, madr, qwc, tadr);
    dma_transfer(ch, chcr, madr, qwc, tadr);
    reg(base) &= ~0x100u;           // STR off: transfer finished
    reg(base + 0x20) = 0;
    reg(D_STAT) |= 1u << ch;        // channel interrupt status
    kernel_dmac_event(ch);
}

uint32_t hw_read32(uint32_t a) {
    uint32_t p = a & 0x1FFFFFFFu;
    if (p >= 0x12000000u) {
        uint32_t off = p & 0x1FFF;
        uint64_t v = g_gs_priv[off >> 3];
        if ((off & ~7u) == (GS_CSR & 0x1FFF)) {
            kernel_safepoint();
            // toggle FIELD and report VSINT so vsync polls make progress
            static uint32_t field;
            field ^= 1;
            v = (v & ~0x200Bull) | (field ? 0x2000 : 0) | 0x8 | gs_csr_events();
        }
        return (uint32_t)(v >> ((off & 4) * 8));
    }
    { int ti = timer_index(p); if (ti >= 0 && p != 0x10001810) return timer_read(ti, p & 0x3F); }
    if (p == 0x10001810 && g_tim[3].mode) return timer_read(3, 0x10);
    switch (p) {
    case 0x10001810:  // T3 MODE: reported "running" (see hw_init) until the game programs it
        if (!g_tim[3].mode) return 0x100;
        break;
    case VIF0_STAT: case VIF1_STAT: case GIF_STAT: return 0;
    case INTC_STAT: kernel_safepoint(); break;
    }
    if (dma_channel(p) >= 0 && (p & 0xFF) == 0) kernel_safepoint();
    uint32_t v = reg(p);
    if (rt_once(0x200000000ull | p)) rt_log(LOG_INFO, "hw read  %08X -> %08X", p, v);
    return v;
}

void hw_write32(uint32_t a, uint32_t v) {
    uint32_t p = a & 0x1FFFFFFFu;
    if (p >= 0x12000000u) {
        uint32_t off = p & 0x1FFF;
        uint64_t &r = g_gs_priv[off >> 3];
        uint64_t sh = (off & 4) * 8;
        if ((off & ~7u) == (GS_CSR & 0x1FFF)) { gs_csr_write(v); return; }
        r = (r & ~(0xFFFFFFFFull << sh)) | ((uint64_t)v << sh);
        gs_priv_write(off & ~7u, r);
        return;
    }
    if (rt_once(0x300000000ull | p)) rt_log(LOG_INFO, "hw write %08X <- %08X", p, v);
    { int ti = timer_index(p); if (ti >= 0) { timer_write(ti, p & 0x3F, v); return; } }
    switch (p) {
    case D_STAT: {
        uint32_t s = reg(D_STAT);
        s &= ~(v & 0xFFFF);              // clear status bits
        s ^= (v & 0xFFFF0000u);          // toggle mask bits
        reg(D_STAT) = s;
        return;
    }
    case INTC_STAT: reg(INTC_STAT) &= ~v; return;
    case INTC_MASK: reg(INTC_MASK) ^= v; return;
    case SIF_MSFLG: reg(SIF_MSFLG) |= v; sif_flag_write(p, v); return;
    case SIF_SMFLG: reg(SIF_SMFLG) &= ~v; return;
    case D_ENABLEW: reg(D_ENABLER) = v; reg(D_ENABLEW) = v; return;
    }
    int ch = dma_channel(p);
    reg(p) = v;
    if (ch >= 0 && (p & 0xFF) == 0 && (v & 0x100)) dma_start(ch);
    if (p >= GIF_FIFO && p < GIF_FIFO + 0x10) gif_fifo_write32(v);
}

uint64_t hw_read64(uint32_t a) {
    uint32_t p = a & 0x1FFFFFFFu;
    if (p >= 0x12000000u) return (uint64_t)hw_read32(a) | ((uint64_t)hw_read32(a + 4) << 32);
    return (uint64_t)hw_read32(a) | ((uint64_t)hw_read32(a + 4) << 32);
}

void hw_write64(uint32_t a, uint64_t v) {
    uint32_t p = a & 0x1FFFFFFFu;
    if (p >= 0x12000000u) {
        uint32_t off = p & 0x1FFF;
        if ((off & ~7u) == (GS_CSR & 0x1FFF)) { gs_csr_write((uint32_t)v); return; }
        g_gs_priv[off >> 3] = v;
        gs_priv_write(off & ~7u, v);
        return;
    }
    hw_write32(a, (uint32_t)v);
    hw_write32(a + 4, (uint32_t)(v >> 32));
}

uint32_t hw_reg(uint32_t p) { return reg(p); }
void hw_set_reg(uint32_t p, uint32_t v) { reg(p) = v; }
void hw_raise_intc(int line) { reg(INTC_STAT) |= 1u << line; }
