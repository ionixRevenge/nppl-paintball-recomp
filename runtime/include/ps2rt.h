/* PS2 EE runtime interface used by recompiled code. */
#pragma once
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef union R128 {
    uint64_t ud[2]; int64_t sd[2];
    uint32_t uw[4]; int32_t sw[4];
    uint16_t uh[8]; int16_t sh[8];
    uint8_t  ub[16]; int8_t sb[16];
    float    f[4];
} R128;

typedef union FPR { float f; uint32_t u; int32_t s; } FPR;

typedef struct R5900 {
    R128 r[32];
    R128 hi, lo;            /* ud[0] = pipeline 0, ud[1] = pipeline 1 (MULT1/DIV1) */
    uint32_t sa;
    uint32_t fcr31;
    FPR f[32];
    float acc;
    /* VU0 (macro mode) */
    R128 vf[32];
    union { uint32_t vi[32]; float vi_f[32]; };
    R128 vacc;
    float vq;
    /* COP0 */
    uint32_t cop0[32];
    uint64_t cycles;
    int thread_id;
    uint32_t pc;            /* last function entered / loop target (diagnostics) */
} R5900;

typedef void (*RtFunc)(R5900 *c);
typedef struct RtFuncEntry { uint32_t addr; RtFunc fn; } RtFuncEntry;
extern const RtFuncEntry rt_func_table[];
extern const unsigned rt_func_count;

#define SX32(x) ((uint64_t)(int64_t)(int32_t)(uint32_t)(x))
#define FCR_C (1u << 23)
#define RAM_SIZE 0x02000000u

extern uint8_t *g_ram;      /* 32 MB main RAM */
extern uint8_t g_spr[0x4000];   /* scratchpad, 0x70000000 */

static const R128 VF0 = { { 0, 0x3F80000000000000ull } };

/* ---- memory: fast path for main RAM and scratchpad, slow path for the rest */
uint8_t  rt_read8 (uint32_t a);
uint16_t rt_read16(uint32_t a);
uint32_t rt_read32(uint32_t a);
uint64_t rt_read64(uint32_t a);
void     rt_read128(uint32_t a, R128 *v);
void rt_write8 (uint32_t a, uint8_t v);
void rt_write16(uint32_t a, uint16_t v);
void rt_write32(uint32_t a, uint32_t v);
void rt_write64(uint32_t a, uint64_t v);
void rt_write128(uint32_t a, const R128 *v);

static inline uint8_t *rt_fast(uint32_t a, uint32_t n) {
    if (a < RAM_SIZE) return g_ram + a;
    if ((a & 0xFFFFC000u) == 0x70000000u) return g_spr + (a & 0x3FFF);
    (void)n;
    return 0;
}
#define RT_LOAD(T, a, slow) do { uint8_t *_p = rt_fast((a), sizeof(T)); if (_p) { T _v; memcpy(&_v, _p, sizeof(T)); return _v; } return slow(a); } while (0)
static inline uint8_t  LB(uint32_t a) { RT_LOAD(uint8_t,  a, rt_read8); }
static inline uint16_t LH(uint32_t a) { RT_LOAD(uint16_t, a, rt_read16); }
static inline uint32_t LW(uint32_t a) { RT_LOAD(uint32_t, a, rt_read32); }
static inline uint64_t LD(uint32_t a) { RT_LOAD(uint64_t, a, rt_read64); }
static inline void LQ(R128 *v, uint32_t a) {
    a &= ~15u; uint8_t *p = rt_fast(a, 16);
    if (p) memcpy(v, p, 16); else rt_read128(a, v);
}
#define RT_STORE(T, a, v, slow) do { uint8_t *_p = rt_fast((a), sizeof(T)); if (_p) { T _v = (v); memcpy(_p, &_v, sizeof(T)); } else slow((a), (v)); } while (0)
static inline void SB(uint32_t a, uint8_t v)  { RT_STORE(uint8_t,  a, v, rt_write8); }
static inline void SH(uint32_t a, uint16_t v) { RT_STORE(uint16_t, a, v, rt_write16); }
static inline void SW(uint32_t a, uint32_t v) { RT_STORE(uint32_t, a, v, rt_write32); }
static inline void SD(uint32_t a, uint64_t v) { RT_STORE(uint64_t, a, v, rt_write64); }
static inline void SQ(uint32_t a, const R128 *v) {
    a &= ~15u; uint8_t *p = rt_fast(a, 16);
    if (p) memcpy(p, v, 16); else rt_write128(a, v);
}

/* ---- PS2 FPU semantics (no inf/nan; divide by zero saturates) */
static inline float fclamp(float x) {
    if (isnan(x)) return signbit(x) ? -FLT_MAX : FLT_MAX;
    if (x > FLT_MAX) return FLT_MAX;
    if (x < -FLT_MAX) return -FLT_MAX;
    return x;
}
static inline float fdiv_ps2(R5900 *c, float a, float b) {
    (void)c;
    if (b == 0.0f) return (signbit(a) != signbit(b)) ? -FLT_MAX : FLT_MAX;
    return fclamp(a / b);
}
static inline int32_t cvt_w_s(float x) {
    if (x >= 2147483648.0f) return 0x7FFFFFFF;
    if (x <= -2147483648.0f) return (int32_t)0x80000000;
    if (isnan(x)) return 0x7FFFFFFF;
    return (int32_t)x;
}
static inline int32_t vu_ftoi(float x, int shift) { return cvt_w_s(x * (float)(1 << shift)); }
/* VU0 macro mode FMAC flags: MAC (vi17: Z bits 0-3, S 4-7, U 8-11, O 12-15; x = highest bit of
   each group) and STATUS (vi16: Z S U O in bits 0-3 plus sticky copies in 6-9), from the clamped
   results of the fields in dest. Game code (Havok collision) tests vector signs this way.
   As on the VU (and PCSX2), S is the result's sign bit, so -0 sets both Z and S, and a
   denormal result counts as zero (Z + U). The x86 sum/product gives -0 whenever a 0 meets a
   negative operand, which axis-aligned normals do constantly. vu0_flags_legacy (env
   NPPL_VUFLAGS_OLD=1) restores the old behaviour (-0 not negative) for A/B tests. */
extern int vu0_flags_legacy;
static inline void vu0_macro_flags(R5900 *c, int dest, const float *r) {
    uint32_t mac = 0;
    for (int k = 0; k < 4; k++) {
        if (!(dest & (8 >> k))) continue;
        int bit = 3 - k;
        uint32_t u; memcpy(&u, &r[k], 4);
        uint32_t ex = u & 0x7F800000u;
        if (vu0_flags_legacy) {
            if (r[k] == 0.0f) mac |= 1u << bit;
            else if (u >> 31) mac |= 1u << (bit + 4);
        } else {
            if (u >> 31) mac |= 1u << (bit + 4);
            if (ex == 0) mac |= ((u & 0x7FFFFFFFu) ? 0x101u : 0x001u) << bit;
        }
        if (fabsf(r[k]) >= FLT_MAX) mac |= 1u << (bit + 12);
    }
    uint32_t st = c->vi[16] & 0xFF0;
    if (mac & 0x000F) st |= 0x001 | 0x040;
    if (mac & 0x00F0) st |= 0x002 | 0x080;
    if (mac & 0x0F00) st |= 0x004 | 0x100;
    if (mac & 0xF000) st |= 0x008 | 0x200;
    c->vi[17] = mac;
    c->vi[16] = st;
}

/* ---- runtime services */
extern uint32_t rt_loop_counter;
void rt_loop_hook(R5900 *c, uint32_t target);
#define RT_BACK(c, t) do { if (++rt_loop_counter >= 0x10000u) rt_loop_hook((c), (t)); } while (0)

void rt_call(R5900 *c, uint32_t addr);
void *rt_jmpbuf(uint32_t guest_buf);   /* host jmp_buf paired with a guest jmp_buf */
void rt_bad_entry(R5900 *c, uint32_t addr);
void rt_syscall(R5900 *c, uint32_t pc);
void rt_break(R5900 *c, uint32_t pc);
void rt_invalid(R5900 *c, uint32_t pc, uint32_t word);
void rt_unimpl(R5900 *c, uint32_t pc, const char *op);
void rt_ei(R5900 *c);
void rt_di(R5900 *c);
void rt_eret(R5900 *c);
uint32_t rt_mfc0(R5900 *c, int reg);
void rt_mtc0(R5900 *c, int reg, uint32_t v);
uint32_t rt_mfperf(R5900 *c, int sel);
void rt_mtperf(R5900 *c, int sel, uint32_t v);
void rt_div(R5900 *c, int32_t a, int32_t b, int pipe);
void rt_divu(R5900 *c, uint32_t a, uint32_t b, int pipe);
void rt_lwl(R5900 *c, int rt, uint32_t a);
void rt_lwr(R5900 *c, int rt, uint32_t a);
void rt_ldl(R5900 *c, int rt, uint32_t a);
void rt_ldr(R5900 *c, int rt, uint32_t a);
void rt_swl(R5900 *c, int rt, uint32_t a);
void rt_swr(R5900 *c, int rt, uint32_t a);
void rt_sdl(R5900 *c, int rt, uint32_t a);
void rt_sdr(R5900 *c, int rt, uint32_t a);
uint32_t rt_cfc2(R5900 *c, int reg);
void rt_ctc2(R5900 *c, int reg, uint32_t v);
void rt_vcallms(R5900 *c, uint32_t addr);
void rt_vsqi(R5900 *c, int dest, int fs, int it);
void rt_vlqi(R5900 *c, int dest, int ft, int is);

/* ---- MMI (128-bit multimedia) */
#define MMI(name) void mmi_##name(R5900 *c, int rd, int rs, int rt, int sa)
MMI(paddh); MMI(paddw); MMI(paddub); MMI(padduw); MMI(psubb); MMI(psubw);
MMI(pcgth); MMI(pminh); MMI(pmaxh);
MMI(psllh); MMI(psrlh); MMI(psrah); MMI(psraw);
MMI(pand); MMI(por); MMI(pxor); MMI(pnor);
MMI(pcpyld); MMI(pcpyud); MMI(pcpyh);
MMI(pextlb); MMI(pextub); MMI(pextlw); MMI(pextuw);
MMI(ppacb); MMI(pexew); MMI(prot3w); MMI(plzcw);
MMI(qfsrv);
#undef MMI

#ifdef __cplusplus
}
#endif
