// CPU helper operations for recompiled R5900 code.
#include "ps2rt.h"
#include "rt_internal.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

template <typename T> static T sat_add_u(T a, T b) { T r = (T)(a + b); return r < a ? (T)~(T)0 : r; }

extern "C" {

void rt_div(R5900 *c, int32_t a, int32_t b, int p) {
    int32_t q, r;
    if (b == 0) { q = a >= 0 ? -1 : 1; r = a; }
    else if (a == INT32_MIN && b == -1) { q = INT32_MIN; r = 0; }
    else { q = a / b; r = a % b; }
    c->lo.ud[p] = SX32(q); c->hi.ud[p] = SX32(r);
}
void rt_divu(R5900 *c, uint32_t a, uint32_t b, int p) {
    uint32_t q, r;
    if (b == 0) { q = 0xFFFFFFFFu; r = a; } else { q = a / b; r = a % b; }
    c->lo.ud[p] = SX32(q); c->hi.ud[p] = SX32(r);
}

// ---- unaligned loads/stores (little-endian)
void rt_lwl(R5900 *c, int rt, uint32_t a) {
    uint32_t sh = (a & 3) * 8, mem = LW(a & ~3u);
    uint32_t v = ((uint32_t)c->r[rt].ud[0] & (0x00FFFFFFu >> sh)) | (mem << (24 - sh));
    if (rt) c->r[rt].ud[0] = SX32(v);
}
void rt_lwr(R5900 *c, int rt, uint32_t a) {
    uint32_t sh = (a & 3) * 8, mem = LW(a & ~3u);
    if (!rt) return;
    if (sh == 0) { c->r[rt].ud[0] = SX32(mem); return; }
    uint32_t v = ((uint32_t)c->r[rt].ud[0] & (0xFFFFFF00u << (24 - sh))) | (mem >> sh);
    c->r[rt].ud[0] = (c->r[rt].ud[0] & 0xFFFFFFFF00000000ull) | v;
}
void rt_ldl(R5900 *c, int rt, uint32_t a) {
    uint32_t sh = (a & 7) * 8; uint64_t mem = LD(a & ~7u);
    if (rt) c->r[rt].ud[0] = (c->r[rt].ud[0] & (0x00FFFFFFFFFFFFFFull >> sh)) | (mem << (56 - sh));
}
void rt_ldr(R5900 *c, int rt, uint32_t a) {
    uint32_t sh = (a & 7) * 8; uint64_t mem = LD(a & ~7u);
    if (rt) c->r[rt].ud[0] = (c->r[rt].ud[0] & (0xFFFFFFFFFFFFFF00ull << (56 - sh))) | (mem >> sh);
}
void rt_swl(R5900 *c, int rt, uint32_t a) {
    uint32_t sh = (a & 3) * 8, mem = LW(a & ~3u), r = (uint32_t)c->r[rt].ud[0];
    SW(a & ~3u, (mem & (0xFFFFFF00u << sh)) | (r >> (24 - sh)));
}
void rt_swr(R5900 *c, int rt, uint32_t a) {
    uint32_t sh = (a & 3) * 8, mem = LW(a & ~3u), r = (uint32_t)c->r[rt].ud[0];
    SW(a & ~3u, (mem & (0x00FFFFFFu >> (24 - sh))) | (r << sh));
}
void rt_sdl(R5900 *c, int rt, uint32_t a) {
    uint32_t sh = (a & 7) * 8; uint64_t mem = LD(a & ~7u), r = c->r[rt].ud[0];
    SD(a & ~7u, (mem & (0xFFFFFFFFFFFFFF00ull << sh)) | (r >> (56 - sh)));
}
void rt_sdr(R5900 *c, int rt, uint32_t a) {
    uint32_t sh = (a & 7) * 8; uint64_t mem = LD(a & ~7u), r = c->r[rt].ud[0];
    SD(a & ~7u, (mem & (0x00FFFFFFFFFFFFFFull >> (56 - sh))) | (r << sh));
}

// ---- VU0 control registers / macro-mode memory access
int vu0_flags_legacy = getenv("NPPL_VUFLAGS_OLD") && atoi(getenv("NPPL_VUFLAGS_OLD")) != 0;

uint32_t rt_cfc2(R5900 *c, int reg) {
    if (reg == 22) { uint32_t u; memcpy(&u, &c->vq, 4); return u; }
    if (reg < 16) return c->vi[reg] & 0xFFFF;
    return c->vi[reg];
}
void rt_ctc2(R5900 *c, int reg, uint32_t v) {
    if (reg == 0) return;
    if (reg == 22) { memcpy(&c->vq, &v, 4); return; }
    if (reg < 16) v &= 0xFFFF;
    if (reg == 28) { /* FBRST: reset/force-break VU0/VU1 */ vu_fbrst(v); }
    c->vi[reg] = v;
}
void rt_vsqi(R5900 *c, int dest, int fs, int it) {
    uint32_t a = (c->vi[it] & 0xFF) * 16;
    for (int k = 0; k < 4; k++)
        if (dest & (8 >> k)) memcpy(g_vu0_mem + a + k * 4, &c->vf[fs].uw[k], 4);
    if (it) c->vi[it] = (c->vi[it] + 1) & 0xFFFF;
}
void rt_vlqi(R5900 *c, int dest, int ft, int is) {
    uint32_t a = (c->vi[is] & 0xFF) * 16;
    if (ft)
        for (int k = 0; k < 4; k++)
            if (dest & (8 >> k)) memcpy(&c->vf[ft].uw[k], g_vu0_mem + a + k * 4, 4);
    if (is) c->vi[is] = (c->vi[is] + 1) & 0xFFFF;
}
void rt_vcallms(R5900 *c, uint32_t addr) {
    vu0_run_micro(c, addr);
}

// ---- MMI
#define MMI(name) void mmi_##name(R5900 *c, int rd, int rs, int rt, int sa)
#define SRC R128 s = c->r[rs], t = c->r[rt]; R128 &d = c->r[rd]; (void)s; (void)t; (void)sa

MMI(paddh)  { SRC; for (int i = 0; i < 8; i++) d.uh[i] = (uint16_t)(s.uh[i] + t.uh[i]); }
MMI(paddw)  { SRC; for (int i = 0; i < 4; i++) d.uw[i] = s.uw[i] + t.uw[i]; }
MMI(paddub) { SRC; for (int i = 0; i < 16; i++) d.ub[i] = sat_add_u<uint8_t>(s.ub[i], t.ub[i]); }
MMI(padduw) { SRC; for (int i = 0; i < 4; i++) d.uw[i] = sat_add_u<uint32_t>(s.uw[i], t.uw[i]); }
MMI(psubb)  { SRC; for (int i = 0; i < 16; i++) d.ub[i] = (uint8_t)(s.ub[i] - t.ub[i]); }
MMI(psubw)  { SRC; for (int i = 0; i < 4; i++) d.uw[i] = s.uw[i] - t.uw[i]; }
MMI(pcgth)  { SRC; for (int i = 0; i < 8; i++) d.uh[i] = s.sh[i] > t.sh[i] ? 0xFFFF : 0; }
MMI(pminh)  { SRC; for (int i = 0; i < 8; i++) d.sh[i] = std::min(s.sh[i], t.sh[i]); }
MMI(pmaxh)  { SRC; for (int i = 0; i < 8; i++) d.sh[i] = std::max(s.sh[i], t.sh[i]); }
MMI(psllh)  { SRC; for (int i = 0; i < 8; i++) d.uh[i] = (uint16_t)(t.uh[i] << (sa & 15)); }
MMI(psrlh)  { SRC; for (int i = 0; i < 8; i++) d.uh[i] = (uint16_t)(t.uh[i] >> (sa & 15)); }
MMI(psrah)  { SRC; for (int i = 0; i < 8; i++) d.sh[i] = (int16_t)(t.sh[i] >> (sa & 15)); }
MMI(psraw)  { SRC; for (int i = 0; i < 4; i++) d.sw[i] = t.sw[i] >> (sa & 31); }
MMI(pand)   { SRC; d.ud[0] = s.ud[0] & t.ud[0]; d.ud[1] = s.ud[1] & t.ud[1]; }
MMI(por)    { SRC; d.ud[0] = s.ud[0] | t.ud[0]; d.ud[1] = s.ud[1] | t.ud[1]; }
MMI(pxor)   { SRC; d.ud[0] = s.ud[0] ^ t.ud[0]; d.ud[1] = s.ud[1] ^ t.ud[1]; }
MMI(pnor)   { SRC; d.ud[0] = ~(s.ud[0] | t.ud[0]); d.ud[1] = ~(s.ud[1] | t.ud[1]); }
MMI(pcpyld) { SRC; d.ud[0] = t.ud[0]; d.ud[1] = s.ud[0]; }
MMI(pcpyud) { SRC; d.ud[0] = s.ud[1]; d.ud[1] = t.ud[1]; }
MMI(pcpyh)  { SRC; for (int i = 0; i < 4; i++) { d.uh[i] = t.uh[0]; d.uh[i + 4] = t.uh[4]; } }
MMI(pextlb) { SRC; for (int i = 0; i < 8; i++) { d.ub[2 * i] = t.ub[i]; d.ub[2 * i + 1] = s.ub[i]; } }
MMI(pextub) { SRC; for (int i = 0; i < 8; i++) { d.ub[2 * i] = t.ub[i + 8]; d.ub[2 * i + 1] = s.ub[i + 8]; } }
MMI(pextlw) { SRC; d.uw[0] = t.uw[0]; d.uw[1] = s.uw[0]; d.uw[2] = t.uw[1]; d.uw[3] = s.uw[1]; }
MMI(pextuw) { SRC; d.uw[0] = t.uw[2]; d.uw[1] = s.uw[2]; d.uw[2] = t.uw[3]; d.uw[3] = s.uw[3]; }
MMI(ppacb)  { SRC; for (int i = 0; i < 8; i++) { d.ub[i] = t.ub[2 * i]; d.ub[i + 8] = s.ub[2 * i]; } }
MMI(pexew)  { SRC; d.uw[0] = t.uw[2]; d.uw[1] = t.uw[1]; d.uw[2] = t.uw[0]; d.uw[3] = t.uw[3]; }
MMI(prot3w) { SRC; d.uw[0] = t.uw[1]; d.uw[1] = t.uw[2]; d.uw[2] = t.uw[0]; d.uw[3] = t.uw[3]; }
MMI(plzcw) {
    SRC;
    for (int i = 0; i < 2; i++) {
        uint32_t v = s.uw[i];
        if (v & 0x80000000u) v = ~v;
        int n = 0;
        while (n < 32 && !(v & (0x80000000u >> n))) n++;
        d.uw[i] = (uint32_t)(n - 1);
    }
}
MMI(qfsrv) {
    SRC;
    uint8_t buf[32];
    memcpy(buf, &t, 16); memcpy(buf + 16, &s, 16);
    memcpy(&d, buf + (c->sa & 15), 16);
}

} // extern "C"
