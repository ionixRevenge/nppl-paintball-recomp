// VU micro-mode interpreter (shared by VU0 and VU1).
//
// Each 64-bit instruction is a pair: lower op (bits 0-31) and upper op (bits 32-63).
// Both halves read the registers as they were before the pair; the upper result is
// committed last (so it wins if both write the same register). Q and P are pipelined:
// DIV/SQRT/RSQRT and EFU results become visible only after their latency (or WAITQ/WAITP),
// which microcode relies on when it software-pipelines vertex loops.
#include "rt_internal.h"
#include "vu.h"
#include <cmath>
#include <chrono>
#include <immintrin.h>

VuState g_vu0, g_vu1;
static R5900 g_vu1_regs;

enum { VI_STATUS = 16, VI_MAC = 17, VI_CLIP = 18, VI_R = 20, VI_I = 21, VI_Q = 22, VI_TPC = 26 };

void vu_init() {
    g_vu1 = VuState{};
    memset(&g_vu1_regs, 0, sizeof g_vu1_regs);
    g_vu1_regs.vf[0] = VF0;
    g_vu1.regs = &g_vu1_regs;
    g_vu1.mem = g_vu1_mem; g_vu1.mem_mask = sizeof g_vu1_mem - 1;
    g_vu1.micro = g_vu1_micro; g_vu1.micro_mask = sizeof g_vu1_micro - 1;
    g_vu1.index = 1;
    g_vu0 = VuState{};
    g_vu0.mem = g_vu0_mem; g_vu0.mem_mask = sizeof g_vu0_mem - 1;
    g_vu0.micro = g_vu0_micro; g_vu0.micro_mask = sizeof g_vu0_micro - 1;
    g_vu0.index = 0;
}

static inline float vclamp(float x) {
    if (std::isnan(x)) return std::signbit(x) ? -FLT_MAX : FLT_MAX;
    if (x > FLT_MAX) return FLT_MAX;
    if (x < -FLT_MAX) return -FLT_MAX;
    if (x != 0.0f && std::fabs(x) < FLT_MIN) return std::signbit(x) ? -0.0f : 0.0f;  // no denormals
    return x;
}

static inline float rdf(const R128 &v, int k) { return v.f[k]; }

// MAX/MINI compare the raw bits as sign-magnitude integers and return one operand unchanged
// (no NaN handling), so microcode can use them to move integer data such as GIF tags.
static inline int64_t sm_key(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    int64_t m = u & 0x7FFFFFFF;
    return (u & 0x80000000u) ? -m : m;
}
static inline float vu_max(float a, float b) { return sm_key(a) >= sm_key(b) ? a : b; }
static inline float vu_min(float a, float b) { return sm_key(a) <= sm_key(b) ? a : b; }

struct Ctx {
    VuState &vu;
    R5900 *r;
    uint32_t *vi() { return r->vi; }
};

// MAC flags as on the VU (and PCSX2): S is the result's sign bit, so -0 sets Z and S, and a
// denormal result counts as zero (Z + U). Havok's VU0 solver microcode branches on these.
// vu0_flags_legacy (NPPL_VUFLAGS_OLD=1, ps2rt.h) restores the old "-0 is not negative" rule.
static void set_mac(R5900 *r, int dest, const float *res) {
    uint32_t mac = 0, st = r->vi[VI_STATUS] & 0xFF0;
    for (int k = 0; k < 4; k++) {
        if (!(dest & (8 >> k))) continue;
        int bit = 3 - k;
        if (vu0_flags_legacy) {
            if (res[k] == 0.0f) mac |= 1u << bit;
            if (std::signbit(res[k]) && res[k] != 0.0f) mac |= 1u << (bit + 4);
        } else {
            uint32_t u; memcpy(&u, &res[k], 4);
            if (u >> 31) mac |= 1u << (bit + 4);
            if ((u & 0x7F800000u) == 0) mac |= ((u & 0x7FFFFFFFu) ? 0x101u : 0x001u) << bit;
        }
        if (std::fabs(res[k]) >= FLT_MAX) mac |= 1u << (bit + 12);
    }
    if (mac & 0x000F) st |= 0x001 | 0x040;
    if (mac & 0x00F0) st |= 0x002 | 0x080;
    if (mac & 0x0F00) st |= 0x004 | 0x100;
    if (mac & 0xF000) st |= 0x008 | 0x200;
    r->vi[VI_MAC] = mac;
    r->vi[VI_STATUS] = (r->vi[VI_STATUS] & 0x030) | st | (r->vi[VI_STATUS] & 0xC00);
}

static void commit_q(VuState &vu, bool force) {
    if (vu.q_ready && (force || vu.cycle >= vu.q_ready)) { vu.regs->vq = vu.q_pending; vu.q_ready = 0; }
}
static void commit_p(VuState &vu, bool force) {
    if (vu.p_ready && (force || vu.cycle >= vu.p_ready)) { vu.p = vu.p_pending; vu.p_ready = 0; }
}

// ---------------------------------------------------------------- upper
struct UpperResult { int kind = 0; int reg = 0; int dest = 0; R128 v{}; bool flags = false; int clip = -1; };  // kind 1 = VF, 2 = ACC; clip = new CLIP flags

static void upper(VuState &vu, uint32_t w, UpperResult &out) {
    R5900 *r = vu.regs;
    int dest = (w >> 21) & 15, ft = (w >> 16) & 31, fs = (w >> 11) & 31, fd = (w >> 6) & 31, bc = w & 3;
    const R128 S = r->vf[fs], T = r->vf[ft], A = r->vacc;
    float I; memcpy(&I, &r->vi[VI_I], 4);
    float Q = r->vq;
    int op = w & 63;
    enum { ADD, SUB, MADD, MSUB, MAX, MIN, MUL };
    auto arith = [&](int kind, int src, bool acc, int target) {   // src: 0 = vector, 1 = bc, 2 = q, 3 = i
        float res[4];
        for (int k = 0; k < 4; k++) {
            float a = S.f[k];
            float b = src == 0 ? T.f[k] : src == 1 ? T.f[bc] : src == 2 ? Q : I;
            float v;
            switch (kind) {
            case ADD: v = a + b; break;
            case SUB: v = a - b; break;
            case MADD: v = A.f[k] + a * b; break;
            case MSUB: v = A.f[k] - a * b; break;
            case MAX: res[k] = vu_max(a, b); continue;
            case MIN: res[k] = vu_min(a, b); continue;
            default: v = a * b; break;
            }
            res[k] = vclamp(v);
        }
        out.kind = acc ? 2 : 1; out.reg = target; out.dest = dest;
        out.flags = kind != MAX && kind != MIN;
        for (int k = 0; k < 4; k++) out.v.f[k] = res[k];
    };
    if (op < 0x3C) {
        switch (op) {
        case 0x00: case 0x01: case 0x02: case 0x03: arith(ADD, 1, false, fd); return;
        case 0x04: case 0x05: case 0x06: case 0x07: arith(SUB, 1, false, fd); return;
        case 0x08: case 0x09: case 0x0A: case 0x0B: arith(MADD, 1, false, fd); return;
        case 0x0C: case 0x0D: case 0x0E: case 0x0F: arith(MSUB, 1, false, fd); return;
        case 0x10: case 0x11: case 0x12: case 0x13: arith(MAX, 1, false, fd); return;
        case 0x14: case 0x15: case 0x16: case 0x17: arith(MIN, 1, false, fd); return;
        case 0x18: case 0x19: case 0x1A: case 0x1B: arith(MUL, 1, false, fd); return;
        case 0x1C: arith(MUL, 2, false, fd); return;
        case 0x1D: arith(MAX, 3, false, fd); return;
        case 0x1E: arith(MUL, 3, false, fd); return;
        case 0x1F: arith(MIN, 3, false, fd); return;
        case 0x20: arith(ADD, 2, false, fd); return;
        case 0x21: arith(MADD, 2, false, fd); return;
        case 0x22: arith(ADD, 3, false, fd); return;
        case 0x23: arith(MADD, 3, false, fd); return;
        case 0x24: arith(SUB, 2, false, fd); return;
        case 0x25: arith(MSUB, 2, false, fd); return;
        case 0x26: arith(SUB, 3, false, fd); return;
        case 0x27: arith(MSUB, 3, false, fd); return;
        case 0x28: arith(ADD, 0, false, fd); return;
        case 0x29: arith(MADD, 0, false, fd); return;
        case 0x2A: arith(MUL, 0, false, fd); return;
        case 0x2B: arith(MAX, 0, false, fd); return;
        case 0x2C: arith(SUB, 0, false, fd); return;
        case 0x2D: arith(MSUB, 0, false, fd); return;
        case 0x2E: {  // OPMSUB
            out.kind = 1; out.reg = fd; out.dest = 0xE; out.flags = true;
            out.v.f[0] = vclamp(A.f[0] - S.f[1] * T.f[2]);
            out.v.f[1] = vclamp(A.f[1] - S.f[2] * T.f[0]);
            out.v.f[2] = vclamp(A.f[2] - S.f[0] * T.f[1]);
            return;
        }
        case 0x2F: arith(MIN, 0, false, fd); return;
        default:
            if (rt_once(0xB00000000ull | op)) rt_log(LOG_WARN, "VU%d: unknown upper op %02X (%08X)", vu.index, op, w);
            return;
        }
    }
    int sop = ((w >> 6) & 31) * 4 + (w & 3);
    switch (sop) {
    case 0x00: case 0x01: case 0x02: case 0x03: arith(ADD, 1, true, 0); return;
    case 0x04: case 0x05: case 0x06: case 0x07: arith(SUB, 1, true, 0); return;
    case 0x08: case 0x09: case 0x0A: case 0x0B: arith(MADD, 1, true, 0); return;
    case 0x0C: case 0x0D: case 0x0E: case 0x0F: arith(MSUB, 1, true, 0); return;
    case 0x10: case 0x11: case 0x12: case 0x13: {  // ITOF0/4/12/15
        static const float sc[4] = {1.0f, 1.0f / 16, 1.0f / 4096, 1.0f / 32768};
        out.kind = 1; out.reg = ft; out.dest = dest;
        for (int k = 0; k < 4; k++) out.v.f[k] = (float)S.sw[k] * sc[sop & 3];
        return;
    }
    case 0x14: case 0x15: case 0x16: case 0x17: {  // FTOI0/4/12/15
        static const int sh[4] = {0, 4, 12, 15};
        out.kind = 1; out.reg = ft; out.dest = dest;
        for (int k = 0; k < 4; k++) out.v.sw[k] = vu_ftoi(S.f[k], sh[sop & 3]);
        return;
    }
    case 0x18: case 0x19: case 0x1A: case 0x1B: arith(MUL, 1, true, 0); return;
    case 0x1C: arith(MUL, 2, true, 0); return;
    case 0x1D:  // ABS
        out.kind = 1; out.reg = ft; out.dest = dest;
        for (int k = 0; k < 4; k++) out.v.f[k] = std::fabs(S.f[k]);
        return;
    case 0x1E: arith(MUL, 3, true, 0); return;
    case 0x1F: {  // CLIP
        float wv = std::fabs(T.f[3]);
        uint32_t cf = 0;
        if (S.f[0] > wv) cf |= 0x01; if (S.f[0] < -wv) cf |= 0x02;
        if (S.f[1] > wv) cf |= 0x04; if (S.f[1] < -wv) cf |= 0x08;
        if (S.f[2] > wv) cf |= 0x10; if (S.f[2] < -wv) cf |= 0x20;
        out.clip = (int)(((r->vi[VI_CLIP] << 6) | cf) & 0xFFFFFF);   // committed after the paired lower op
        return;
    }
    case 0x20: arith(ADD, 2, true, 0); return;
    case 0x21: arith(MADD, 2, true, 0); return;
    case 0x22: arith(ADD, 3, true, 0); return;
    case 0x23: arith(MADD, 3, true, 0); return;
    case 0x24: arith(SUB, 2, true, 0); return;
    case 0x25: arith(MSUB, 2, true, 0); return;
    case 0x26: arith(SUB, 3, true, 0); return;
    case 0x27: arith(MSUB, 3, true, 0); return;
    case 0x28: arith(ADD, 0, true, 0); return;
    case 0x29: arith(MADD, 0, true, 0); return;
    case 0x2A: arith(MUL, 0, true, 0); return;
    case 0x2C: arith(SUB, 0, true, 0); return;
    case 0x2D: arith(MSUB, 0, true, 0); return;
    case 0x2E:  // OPMULA
        out.kind = 2; out.dest = 0xE; out.flags = true;
        out.v.f[0] = vclamp(S.f[1] * T.f[2]);
        out.v.f[1] = vclamp(S.f[2] * T.f[0]);
        out.v.f[2] = vclamp(S.f[0] * T.f[1]);
        return;
    case 0x2F: return;  // NOP
    default:
        if (rt_once(0xB10000000ull | sop)) rt_log(LOG_WARN, "VU%d: unknown upper special op %02X (%08X)", vu.index, sop, w);
    }
}

static void commit_upper(VuState &vu, const UpperResult &u) {
    R5900 *r = vu.regs;
    if (u.clip >= 0) r->vi[VI_CLIP] = (uint32_t)u.clip;
    if (!u.kind) return;
    R128 *t = u.kind == 2 ? &r->vacc : &r->vf[u.reg];
    if (u.kind == 1 && u.reg == 0) { if (u.flags) set_mac(r, u.dest, u.v.f); return; }
    for (int k = 0; k < 4; k++) if (u.dest & (8 >> k)) t->uw[k] = u.v.uw[k];
    if (u.flags) set_mac(r, u.dest, u.v.f);
}

// ---------------------------------------------------------------- lower
static inline uint32_t qaddr(VuState &vu, uint32_t qw) { return (qw * 16) & vu.mem_mask; }

static void write_vf(R5900 *r, int reg, int dest, const R128 &v) {
    if (!reg) return;
    for (int k = 0; k < 4; k++) if (dest & (8 >> k)) r->vf[reg].uw[k] = v.uw[k];
}
static void set_vi(R5900 *r, int reg, uint32_t v) { if (reg && reg < 16) r->vi[reg] = v & 0xFFFF; }

static void efu(VuState &vu, float v, int lat) {
    commit_p(vu, true);
    vu.p_pending = vclamp(v);
    vu.p_ready = vu.cycle + lat;
}

static void qdiv(VuState &vu, float v, int lat) {
    commit_q(vu, true);
    vu.q_pending = vclamp(v);
    vu.q_ready = vu.cycle + lat;
}

// returns: 0 = continue, 1 = branch taken (target in *br)
static int lower(VuState &vu, uint32_t w, uint32_t pc, uint32_t *br) {
    R5900 *r = vu.regs;
    uint32_t *vi = r->vi;
    int op = w >> 25;
    int dest = (w >> 21) & 15, it = (w >> 16) & 31, is = (w >> 11) & 31, id = (w >> 6) & 31;
    int ft = it, fs = is;
    int32_t imm11 = (int32_t)(w << 21) >> 21;
    uint32_t imm15 = ((w >> 10) & 0x7800) | (w & 0x7FF);
    int fsf = (w >> 21) & 3, ftf = (w >> 23) & 3;
    auto VI = [&](int n) -> uint32_t { return n < 16 ? (vi[n] & 0xFFFF) : vi[n]; };
    auto branch = [&](uint32_t target_instr) { *br = (target_instr * 8) & vu.micro_mask; return 1; };
    uint32_t next_instr = pc / 8 + 1;
    switch (op) {
    case 0x00: {  // LQ
        R128 v; memcpy(&v, vu.mem + qaddr(vu, (int16_t)VI(is) + imm11), 16);
        write_vf(r, ft, dest, v); return 0;
    }
    case 0x01: {  // SQ
        uint8_t *p = vu.mem + qaddr(vu, (int16_t)VI(it) + imm11);
        for (int k = 0; k < 4; k++) if (dest & (8 >> k)) memcpy(p + 4 * k, &r->vf[fs].uw[k], 4);
        return 0;
    }
    case 0x04: {  // ILW
        uint8_t *p = vu.mem + qaddr(vu, (int16_t)VI(is) + imm11);
        for (int k = 0; k < 4; k++) if (dest & (8 >> k)) { uint16_t v; memcpy(&v, p + 4 * k, 2); set_vi(r, it, v); break; }
        return 0;
    }
    case 0x05: {  // ISW
        uint8_t *p = vu.mem + qaddr(vu, (int16_t)VI(is) + imm11);
        uint32_t v = VI(it);
        for (int k = 0; k < 4; k++) if (dest & (8 >> k)) memcpy(p + 4 * k, &v, 4);
        return 0;
    }
    case 0x08: set_vi(r, it, VI(is) + imm15); return 0;  // IADDIU
    case 0x09: set_vi(r, it, VI(is) - imm15); return 0;  // ISUBIU
    case 0x10: vi[1] = (vi[VI_CLIP] & 0xFFFFFF) == (w & 0xFFFFFF); return 0;               // FCEQ
    case 0x11: vi[VI_CLIP] = w & 0xFFFFFF; return 0;                                       // FCSET
    case 0x12: vi[1] = (vi[VI_CLIP] & w & 0xFFFFFF) != 0; return 0;                        // FCAND
    case 0x13: vi[1] = ((vi[VI_CLIP] | w) & 0xFFFFFF) == 0xFFFFFF; return 0;               // FCOR
    case 0x14: { uint32_t m = ((w >> 10) & 0x800) | (w & 0x7FF); set_vi(r, it, (vi[VI_STATUS] & 0xFFF) == m); return 0; }   // FSEQ
    case 0x15: { uint32_t m = ((w >> 10) & 0x800) | (w & 0x7FF); vi[VI_STATUS] = (vi[VI_STATUS] & 0x3F) | (m & 0xFC0); return 0; }  // FSSET
    case 0x16: { uint32_t m = ((w >> 10) & 0x800) | (w & 0x7FF); set_vi(r, it, vi[VI_STATUS] & m); return 0; }   // FSAND
    case 0x17: { uint32_t m = ((w >> 10) & 0x800) | (w & 0x7FF); set_vi(r, it, (vi[VI_STATUS] | m) & 0xFFF); return 0; }  // FSOR
    case 0x18: set_vi(r, it, (vi[VI_MAC] & 0xFFFF) == VI(is)); return 0;   // FMEQ
    case 0x1A: set_vi(r, it, vi[VI_MAC] & VI(is)); return 0;               // FMAND
    case 0x1B: set_vi(r, it, (vi[VI_MAC] | VI(is)) & 0xFFFF); return 0;    // FMOR
    case 0x1C: set_vi(r, it, vi[VI_CLIP] & 0xFFF); return 0;               // FCGET
    case 0x20: return branch(next_instr + imm11);                          // B
    case 0x21: set_vi(r, it, next_instr + 1); return branch(next_instr + imm11);   // BAL
    case 0x24: return branch(VI(is));                                      // JR
    case 0x25: { uint32_t t = VI(is); set_vi(r, it, next_instr + 1); return branch(t); }  // JALR
    case 0x28: if (VI(is) == VI(it)) return branch(next_instr + imm11); return 0;   // IBEQ
    case 0x29: if (VI(is) != VI(it)) return branch(next_instr + imm11); return 0;   // IBNE
    case 0x2C: if ((int16_t)VI(is) < 0) return branch(next_instr + imm11); return 0;
    case 0x2D: if ((int16_t)VI(is) > 0) return branch(next_instr + imm11); return 0;
    case 0x2E: if ((int16_t)VI(is) <= 0) return branch(next_instr + imm11); return 0;
    case 0x2F: if ((int16_t)VI(is) >= 0) return branch(next_instr + imm11); return 0;
    case 0x40: break;
    default:
        if (rt_once(0xB20000000ull | op)) rt_log(LOG_WARN, "VU%d: unknown lower op %02X (%08X) at %04X", vu.index, op, w, pc);
        return 0;
    }
    // op 0x40: register-form lower ops
    int f = w & 63;
    switch (f) {
    case 0x30: set_vi(r, id, VI(is) + VI(it)); return 0;   // IADD
    case 0x31: set_vi(r, id, VI(is) - VI(it)); return 0;   // ISUB
    case 0x32: set_vi(r, it, VI(is) + ((int32_t)(w << 21) >> 27)); return 0;  // IADDI (imm5 in bits 6-10)
    case 0x34: set_vi(r, id, VI(is) & VI(it)); return 0;   // IAND
    case 0x35: set_vi(r, id, VI(is) | VI(it)); return 0;   // IOR
    case 0x3C: case 0x3D: case 0x3E: case 0x3F: break;
    default:
        if (rt_once(0xB30000000ull | f)) rt_log(LOG_WARN, "VU%d: unknown lower 0x40 op %02X (%08X)", vu.index, f, w);
        return 0;
    }
    int sop = ((w >> 6) & 31) * 4 + (w & 3);
    const R128 S = r->vf[fs];
    switch (sop) {
    case 0x30: write_vf(r, ft, dest, S); return 0;   // MOVE
    case 0x31: { R128 v; v.uw[0] = S.uw[1]; v.uw[1] = S.uw[2]; v.uw[2] = S.uw[3]; v.uw[3] = S.uw[0]; write_vf(r, ft, dest, v); return 0; }  // MR32
    case 0x34: {  // LQI
        R128 v; memcpy(&v, vu.mem + qaddr(vu, VI(is)), 16); write_vf(r, ft, dest, v);
        set_vi(r, is, VI(is) + 1); return 0;
    }
    case 0x35: {  // SQI
        uint8_t *p = vu.mem + qaddr(vu, VI(it));
        for (int k = 0; k < 4; k++) if (dest & (8 >> k)) memcpy(p + 4 * k, &S.uw[k], 4);
        set_vi(r, it, VI(it) + 1); return 0;
    }
    case 0x36: {  // LQD
        set_vi(r, is, VI(is) - 1);
        R128 v; memcpy(&v, vu.mem + qaddr(vu, VI(is)), 16); write_vf(r, ft, dest, v); return 0;
    }
    case 0x37: {  // SQD
        set_vi(r, it, VI(it) - 1);
        uint8_t *p = vu.mem + qaddr(vu, VI(it));
        for (int k = 0; k < 4; k++) if (dest & (8 >> k)) memcpy(p + 4 * k, &S.uw[k], 4);
        return 0;
    }
    case 0x38: {  // DIV
        float a = S.f[fsf], b = r->vf[ft].f[ftf];
        if (b == 0.0f) {
            vi[VI_STATUS] |= (a == 0.0f) ? 0x410 : 0x820;
            qdiv(vu, (std::signbit(a) != std::signbit(b)) ? -FLT_MAX : FLT_MAX, 7);
        } else { vi[VI_STATUS] &= ~0x30u; qdiv(vu, a / b, 7); }
        return 0;
    }
    case 0x39: qdiv(vu, std::sqrt(std::fabs(r->vf[ft].f[ftf])), 7); return 0;   // SQRT
    case 0x3A: {  // RSQRT
        float a = S.f[fsf], b = std::sqrt(std::fabs(r->vf[ft].f[ftf]));
        if (b == 0.0f) { vi[VI_STATUS] |= 0x820; qdiv(vu, std::signbit(a) ? -FLT_MAX : FLT_MAX, 13); }
        else qdiv(vu, a / b, 13);
        return 0;
    }
    case 0x3B: commit_q(vu, true); return 0;   // WAITQ
    case 0x3C: set_vi(r, it, S.uw[fsf] & 0xFFFF); return 0;   // MTIR
    case 0x3D: { R128 v; for (int k = 0; k < 4; k++) v.sw[k] = (int16_t)VI(is); write_vf(r, ft, dest, v); return 0; }   // MFIR
    case 0x3E: {  // ILWR
        uint8_t *p = vu.mem + qaddr(vu, VI(is));
        for (int k = 0; k < 4; k++) if (dest & (8 >> k)) { uint16_t v; memcpy(&v, p + 4 * k, 2); set_vi(r, it, v); break; }
        return 0;
    }
    case 0x3F: {  // ISWR
        uint8_t *p = vu.mem + qaddr(vu, VI(is));
        uint32_t v = VI(it);
        for (int k = 0; k < 4; k++) if (dest & (8 >> k)) memcpy(p + 4 * k, &v, 4);
        return 0;
    }
    case 0x40: case 0x41: {  // RNEXT / RGET
        uint32_t x = vi[VI_R] & 0x7FFFFF;
        if (sop == 0x40) {
            int b = ((x >> 4) ^ (x >> 22)) & 1;
            x = ((x << 1) | b) & 0x7FFFFF;
            vi[VI_R] = x;
        }
        R128 v; for (int k = 0; k < 4; k++) v.uw[k] = 0x3F800000u | x;
        write_vf(r, ft, dest, v); return 0;
    }
    case 0x42: vi[VI_R] = S.uw[fsf] & 0x7FFFFF; return 0;    // RINIT
    case 0x43: vi[VI_R] = (vi[VI_R] ^ S.uw[fsf]) & 0x7FFFFF; return 0;   // RXOR
    case 0x64: { R128 v; for (int k = 0; k < 4; k++) v.f[k] = vu.p; write_vf(r, ft, dest, v); return 0; }   // MFP
    case 0x68: set_vi(r, it, vu.top); return 0;    // XTOP
    case 0x69: set_vi(r, it, vu.itop); return 0;   // XITOP
    case 0x6C:   // XGKICK
        if (vu.index == 1) vu1_xgkick_hook(vu.mem, (VI(is) * 16) & vu.mem_mask, vu.mem_mask + 1);
        return 0;
    case 0x70: efu(vu, S.f[0] * S.f[0] + S.f[1] * S.f[1] + S.f[2] * S.f[2], 11); return 0;              // ESADD
    case 0x71: { float d = S.f[0] * S.f[0] + S.f[1] * S.f[1] + S.f[2] * S.f[2]; efu(vu, d ? 1.0f / d : FLT_MAX, 18); return 0; }   // ERSADD
    case 0x72: efu(vu, std::sqrt(S.f[0] * S.f[0] + S.f[1] * S.f[1] + S.f[2] * S.f[2]), 18); return 0;  // ELENG
    case 0x73: { float d = std::sqrt(S.f[0] * S.f[0] + S.f[1] * S.f[1] + S.f[2] * S.f[2]); efu(vu, d ? 1.0f / d : FLT_MAX, 24); return 0; }  // ERLENG
    case 0x74: efu(vu, S.f[0] != 0 ? std::atan(S.f[1] / S.f[0]) : 0.0f, 54); return 0;   // EATANxy
    case 0x75: efu(vu, S.f[0] != 0 ? std::atan(S.f[2] / S.f[0]) : 0.0f, 54); return 0;   // EATANxz
    case 0x76: efu(vu, S.f[0] + S.f[1] + S.f[2] + S.f[3], 12); return 0;                   // ESUM
    case 0x78: efu(vu, std::sqrt(std::fabs(S.f[fsf])), 12); return 0;                       // ESQRT
    case 0x79: { float d = std::sqrt(std::fabs(S.f[fsf])); efu(vu, d ? 1.0f / d : FLT_MAX, 18); return 0; }   // ERSQRT
    case 0x7A: efu(vu, S.f[fsf] != 0 ? 1.0f / S.f[fsf] : FLT_MAX, 12); return 0;           // ERCPR
    case 0x7B: commit_p(vu, true); return 0;                                                 // WAITP
    case 0x7C: efu(vu, std::sin(S.f[fsf]), 29); return 0;                                   // ESIN
    case 0x7D: efu(vu, std::atan(S.f[fsf]), 54); return 0;                                  // EATAN
    case 0x7E: efu(vu, std::exp(-S.f[fsf]), 44); return 0;                                  // EEXP
    default:
        if (rt_once(0xB40000000ull | sop)) rt_log(LOG_WARN, "VU%d: unknown lower special op %02X (%08X)", vu.index, sop, w);
        return 0;
    }
}


// ---------------------------------------------------------------- pre-decoded lower ops
struct LDec;
typedef int (*LowerFn)(VuState &, const LDec &, uint32_t lo, uint32_t pc, uint32_t *br);
struct LDec { LowerFn fn; uint8_t it, is, id, dest; int32_t imm; int vfw; };

static const __m128i k_dest_mask[16] = {
#define M(d) _mm_setr_epi32((d & 8) ? -1 : 0, (d & 4) ? -1 : 0, (d & 2) ? -1 : 0, (d & 1) ? -1 : 0)
    M(0), M(1), M(2), M(3), M(4), M(5), M(6), M(7), M(8), M(9), M(10), M(11), M(12), M(13), M(14), M(15)
#undef M
};
static __forceinline void put_vf(R5900 *r, int reg, int dest, __m128i v) {
    if (!reg) return;
    __m128i m = k_dest_mask[dest], o = _mm_loadu_si128((const __m128i *)&r->vf[reg]);
    _mm_storeu_si128((__m128i *)&r->vf[reg], _mm_or_si128(_mm_and_si128(m, v), _mm_andnot_si128(m, o)));
}
static __forceinline uint32_t VIr(const R5900 *r, int n) { return r->vi[n] & 0xFFFF; }   // n < 16
static __forceinline void setVI(R5900 *r, int n, uint32_t v) { if (n) r->vi[n] = v & 0xFFFF; }
static __forceinline uint8_t *qptr(VuState &vu, uint32_t qw) { return vu.mem + ((qw * 16) & vu.mem_mask); }

static int l_generic(VuState &vu, const LDec &, uint32_t lo, uint32_t pc, uint32_t *br) { return lower(vu, lo, pc, br); }
static int l_lq(VuState &vu, const LDec &d, uint32_t, uint32_t, uint32_t *) {
    put_vf(vu.regs, d.it, d.dest, _mm_loadu_si128((const __m128i *)qptr(vu, (uint32_t)((int16_t)VIr(vu.regs, d.is) + d.imm)))); return 0;
}
static int l_sq(VuState &vu, const LDec &d, uint32_t, uint32_t, uint32_t *) {
    __m128i *p = (__m128i *)qptr(vu, (uint32_t)((int16_t)VIr(vu.regs, d.it) + d.imm));
    __m128i m = k_dest_mask[d.dest], v = _mm_loadu_si128((const __m128i *)&vu.regs->vf[d.is]), o = _mm_loadu_si128(p);
    _mm_storeu_si128(p, _mm_or_si128(_mm_and_si128(m, v), _mm_andnot_si128(m, o))); return 0;
}
static int l_lqi(VuState &vu, const LDec &d, uint32_t, uint32_t, uint32_t *) {
    R5900 *r = vu.regs; uint32_t a = VIr(r, d.is);
    put_vf(r, d.it, d.dest, _mm_loadu_si128((const __m128i *)qptr(vu, a)));
    setVI(r, d.is, a + 1); return 0;
}
static int l_sqi(VuState &vu, const LDec &d, uint32_t, uint32_t, uint32_t *) {
    R5900 *r = vu.regs; uint32_t a = VIr(r, d.it);
    __m128i *p = (__m128i *)qptr(vu, a);
    __m128i m = k_dest_mask[d.dest], v = _mm_loadu_si128((const __m128i *)&r->vf[d.is]), o = _mm_loadu_si128(p);
    _mm_storeu_si128(p, _mm_or_si128(_mm_and_si128(m, v), _mm_andnot_si128(m, o)));
    setVI(r, d.it, a + 1); return 0;
}
static int l_iaddiu(VuState &vu, const LDec &d, uint32_t, uint32_t, uint32_t *) { setVI(vu.regs, d.it, VIr(vu.regs, d.is) + (uint32_t)d.imm); return 0; }
static int l_isubiu(VuState &vu, const LDec &d, uint32_t, uint32_t, uint32_t *) { setVI(vu.regs, d.it, VIr(vu.regs, d.is) - (uint32_t)d.imm); return 0; }
static int l_iaddi(VuState &vu, const LDec &d, uint32_t, uint32_t, uint32_t *) { setVI(vu.regs, d.it, VIr(vu.regs, d.is) + (uint32_t)d.imm); return 0; }
static int l_iadd(VuState &vu, const LDec &d, uint32_t, uint32_t, uint32_t *) { setVI(vu.regs, d.id, VIr(vu.regs, d.is) + VIr(vu.regs, d.it)); return 0; }
static int l_isub(VuState &vu, const LDec &d, uint32_t, uint32_t, uint32_t *) { setVI(vu.regs, d.id, VIr(vu.regs, d.is) - VIr(vu.regs, d.it)); return 0; }
static int l_iand(VuState &vu, const LDec &d, uint32_t, uint32_t, uint32_t *) { setVI(vu.regs, d.id, VIr(vu.regs, d.is) & VIr(vu.regs, d.it)); return 0; }
static int l_ior(VuState &vu, const LDec &d, uint32_t, uint32_t, uint32_t *) { setVI(vu.regs, d.id, VIr(vu.regs, d.is) | VIr(vu.regs, d.it)); return 0; }
static int l_move(VuState &vu, const LDec &d, uint32_t, uint32_t, uint32_t *) {
    put_vf(vu.regs, d.it, d.dest, _mm_loadu_si128((const __m128i *)&vu.regs->vf[d.is])); return 0;
}
static int l_mtir(VuState &vu, const LDec &d, uint32_t, uint32_t, uint32_t *) { setVI(vu.regs, d.it, vu.regs->vf[d.is].uw[d.imm] & 0xFFFF); return 0; }
static int l_mfir(VuState &vu, const LDec &d, uint32_t, uint32_t, uint32_t *) {
    put_vf(vu.regs, d.it, d.dest, _mm_set1_epi32((int16_t)VIr(vu.regs, d.is))); return 0;
}
static int l_b(VuState &vu, const LDec &d, uint32_t, uint32_t pc, uint32_t *br) { *br = ((pc / 8 + 1 + d.imm) * 8) & vu.micro_mask; return 1; }
static int l_ibeq(VuState &vu, const LDec &d, uint32_t, uint32_t pc, uint32_t *br) {
    if (VIr(vu.regs, d.is) != VIr(vu.regs, d.it)) return 0;
    *br = ((pc / 8 + 1 + d.imm) * 8) & vu.micro_mask; return 1;
}
static int l_ibne(VuState &vu, const LDec &d, uint32_t, uint32_t pc, uint32_t *br) {
    if (VIr(vu.regs, d.is) == VIr(vu.regs, d.it)) return 0;
    *br = ((pc / 8 + 1 + d.imm) * 8) & vu.micro_mask; return 1;
}
static int l_fcand(VuState &vu, const LDec &, uint32_t lo, uint32_t, uint32_t *) { vu.regs->vi[1] = (vu.regs->vi[VI_CLIP] & lo & 0xFFFFFF) != 0; return 0; }
static int l_ilw(VuState &vu, const LDec &d, uint32_t, uint32_t, uint32_t *) {
    const uint8_t *p = qptr(vu, (uint32_t)((int16_t)VIr(vu.regs, d.is) + d.imm));
    for (int k = 0; k < 4; k++) if (d.dest & (8 >> k)) { uint16_t v; memcpy(&v, p + 4 * k, 2); setVI(vu.regs, d.it, v); break; }
    return 0;
}

static int lower_vf_write(uint32_t w) {
    int op = w >> 25, ft = (w >> 16) & 31;
    if (op == 0x00) return ft;                         // LQ
    if (op != 0x40) return (op == 0x01 || op == 0x04 || op == 0x05 || op == 0x08 || op == 0x09 || (op >= 0x10 && op <= 0x1C) ||
                            (op >= 0x20 && op <= 0x2F)) ? 0 : -1;
    int f = w & 63;
    if (f >= 0x30 && f <= 0x35) return 0;               // integer ALU
    if (f < 0x3C) return -1;
    int sop = ((w >> 6) & 31) * 4 + (w & 3);
    switch (sop) {
    case 0x30: case 0x31: case 0x34: case 0x36: case 0x3D: case 0x40: case 0x41: case 0x64: return ft;
    case 0x35: case 0x37: case 0x38: case 0x39: case 0x3A: case 0x3B: case 0x3C: case 0x3E: case 0x3F:
    case 0x42: case 0x43: case 0x68: case 0x69: case 0x6C: case 0x7B: return 0;
    default: return (sop >= 0x70 && sop <= 0x7E) ? 0 : -1;
    }
}

static LDec decode_lower(uint32_t w) {
    LDec d{};
    d.fn = &l_generic;
    d.vfw = lower_vf_write(w);
    int op = w >> 25;
    d.dest = (uint8_t)((w >> 21) & 15); d.it = (uint8_t)((w >> 16) & 31); d.is = (uint8_t)((w >> 11) & 31); d.id = (uint8_t)((w >> 6) & 31);
    int32_t imm11 = (int32_t)(w << 21) >> 21;
    uint32_t imm15 = ((w >> 10) & 0x7800) | (w & 0x7FF);
    bool vi_ok = d.it < 16 && d.is < 16 && d.id < 16;   // integer-register forms only index vi0..vi15
    switch (op) {
    case 0x00: d.imm = imm11; if (d.is < 16) d.fn = &l_lq; return d;
    case 0x01: d.imm = imm11; if (d.it < 16) d.fn = &l_sq; return d;
    case 0x04: d.imm = imm11; if (vi_ok) d.fn = &l_ilw; return d;
    case 0x08: d.imm = (int32_t)imm15; if (vi_ok) d.fn = &l_iaddiu; return d;
    case 0x09: d.imm = (int32_t)imm15; if (vi_ok) d.fn = &l_isubiu; return d;
    case 0x12: d.fn = &l_fcand; return d;
    case 0x20: d.imm = imm11; d.fn = &l_b; return d;
    case 0x28: d.imm = imm11; if (vi_ok) d.fn = &l_ibeq; return d;
    case 0x29: d.imm = imm11; if (vi_ok) d.fn = &l_ibne; return d;
    case 0x40: break;
    default: return d;
    }
    int f = w & 63;
    switch (f) {
    case 0x30: if (vi_ok) d.fn = &l_iadd; return d;
    case 0x31: if (vi_ok) d.fn = &l_isub; return d;
    case 0x32: d.imm = (int32_t)(w << 21) >> 27; if (vi_ok) d.fn = &l_iaddi; return d;
    case 0x34: if (vi_ok) d.fn = &l_iand; return d;
    case 0x35: if (vi_ok) d.fn = &l_ior; return d;
    case 0x3C: case 0x3D: case 0x3E: case 0x3F: break;
    default: return d;
    }
    int sop = ((w >> 6) & 31) * 4 + (w & 3);
    switch (sop) {
    case 0x30: d.fn = &l_move; return d;
    case 0x34: if (d.is < 16) d.fn = &l_lqi; return d;
    case 0x35: if (d.it < 16) d.fn = &l_sqi; return d;
    case 0x3C: d.imm = (w >> 21) & 3; if (d.it < 16) d.fn = &l_mtir; return d;
    case 0x3D: if (d.is < 16) d.fn = &l_mfir; return d;
    default: return d;
    }
}

// ---------------------------------------------------------------- pipeline hazards
struct RegUse { int reg, mask; };
struct HazardInfo { RegUse rd[4], wr[2]; int nr = 0, nw = 0; bool wait_q = false, wait_p = false; };
static inline void add_rd(HazardInfo &h, int reg, int mask) { if (reg && mask && h.nr < 4) h.rd[h.nr++] = {reg, mask}; }
static inline void add_wr(HazardInfo &h, int reg, int mask) { if (reg && mask && h.nw < 2) h.wr[h.nw++] = {reg, mask}; }
static HazardInfo decode_hazards(uint32_t up, uint32_t lo) {
    HazardInfo h;
    int dest = (up >> 21) & 15, ft = (up >> 16) & 31, fs = (up >> 11) & 31, fd = (up >> 6) & 31, bc = up & 3;
    int bcm = 8 >> bc, op = up & 63;
    if (op < 0x3C) {
        if (op < 0x1C) { add_rd(h, fs, dest); add_rd(h, ft, bcm); }
        else if (op < 0x28) add_rd(h, fs, dest);
        else if (op <= 0x2F) { add_rd(h, fs, dest); add_rd(h, ft, op == 0x2E ? 0xE : dest); }
        add_wr(h, fd, op == 0x2E ? 0xE : dest);
    } else {
        int sop = ((up >> 6) & 31) * 4 + (up & 3);
        if (sop < 0x10 || (sop >= 0x18 && sop <= 0x1B)) { add_rd(h, fs, dest); add_rd(h, ft, bcm); }
        else if (sop < 0x18 || sop == 0x1D) { add_rd(h, fs, dest); add_wr(h, ft, dest); }
        else if (sop == 0x1F) { add_rd(h, fs, 0xE); add_rd(h, ft, 1); }
        else if (sop == 0x2E) { add_rd(h, fs, 0xE); add_rd(h, ft, 0xE); }
        else if (sop >= 0x28 && sop <= 0x2D) { add_rd(h, fs, dest); add_rd(h, ft, dest); }
        else if (sop != 0x2F) add_rd(h, fs, dest);
    }
    if (up & 0x80000000u) return h;   // lower word is an immediate
    int lop = lo >> 25, ldest = (lo >> 21) & 15, lft = (lo >> 16) & 31, lfs = (lo >> 11) & 31;
    int fsf = 8 >> ((lo >> 21) & 3), ftf = 8 >> ((lo >> 23) & 3);
    if (lop == 0x00) add_wr(h, lft, ldest);                 // LQ
    else if (lop == 0x01) add_rd(h, lfs, ldest);            // SQ
    else if (lop == 0x40 && (lo & 0x3C) == 0x3C) {
        int lsop = ((lo >> 6) & 31) * 4 + (lo & 3);
        switch (lsop) {
        case 0x30: add_rd(h, lfs, ldest); add_wr(h, lft, ldest); break;   // MOVE
        case 0x31: add_rd(h, lfs, 0xF); add_wr(h, lft, ldest); break;     // MR32
        case 0x34: case 0x36: add_wr(h, lft, ldest); break;             // LQI / LQD
        case 0x35: case 0x37: add_rd(h, lfs, ldest); break;             // SQI / SQD
        case 0x38: case 0x3A: add_rd(h, lfs, fsf); add_rd(h, lft, ftf); h.wait_q = true; break;   // DIV / RSQRT
        case 0x39: add_rd(h, lft, ftf); h.wait_q = true; break;         // SQRT
        case 0x3B: h.wait_q = true; break;                              // WAITQ
        case 0x3C: add_rd(h, lfs, fsf); break;                          // MTIR
        case 0x3D: case 0x40: case 0x41: case 0x64: add_wr(h, lft, ldest); break;   // MFIR / RNEXT / RGET / MFP
        case 0x42: case 0x43: add_rd(h, lfs, fsf); break;               // RINIT / RXOR
        case 0x7B: h.wait_p = true; break;                              // WAITP
        default:
            if (lsop >= 0x70 && lsop <= 0x7E) { add_rd(h, lfs, lsop <= 0x73 ? 0xE : lsop == 0x76 ? 0xF : fsf); h.wait_p = true; }
        }
    }
    return h;
}


// ---------------------------------------------------------------- decoded fast path
// Each micro-memory slot caches its decode (validated against the raw words, so MPG uploads
// need no invalidation): precomputed hazard info plus a compact upper op run with SSE.
enum { UK_SLOW, UK_NOP, UK_ARITH, UK_ITOF, UK_FTOI, UK_ABS };
struct UDec;
typedef void (*UpperFn)(VuState &, const UDec &, uint32_t, UpperResult &);
struct UDec { uint8_t kind, op, src, acc, dest, fs, ft, fd, bc, sh; UpperFn fn; };
typedef void (*UpperDirectFn)(VuState &, const UDec &);
struct VuDec {
    uint32_t lo, up; bool valid; UDec u;
    // hazards flattened to VF field indices (reg * 4 + lane)
    uint8_t nrd, nwr; bool wait_q, wait_p, lower_nop;
    LDec l;
    UpperDirectFn ud;   // non-null: run lower first, then this writes the upper result directly
    uint8_t rd[16], wr[8];
};
static void flatten_hazards(VuDec &d, const HazardInfo &h) {
    d.nrd = d.nwr = 0; d.wait_q = h.wait_q; d.wait_p = h.wait_p;
    for (int i = 0; i < h.nr; i++)
        for (int k = 0; k < 4; k++) if (h.rd[i].mask & (8 >> k)) d.rd[d.nrd++] = (uint8_t)(h.rd[i].reg * 4 + k);
    for (int i = 0; i < h.nw; i++)
        for (int k = 0; k < 4; k++) if (h.wr[i].mask & (8 >> k)) d.wr[d.nwr++] = (uint8_t)(h.wr[i].reg * 4 + k);
}
static VuDec g_dec[2][2048];

static UDec decode_upper(uint32_t w) {
    UDec d{};
    d.dest = (w >> 21) & 15; d.ft = (w >> 16) & 31; d.fs = (w >> 11) & 31; d.fd = (w >> 6) & 31; d.bc = w & 3;
    int op = w & 63;
    enum { ADD, SUB, MADD, MSUB, MAX, MIN, MUL };
    auto A = [&](int kind, int src, bool acc) { d.kind = UK_ARITH; d.op = (uint8_t)kind; d.src = (uint8_t)src; d.acc = acc; };
    if (op < 0x3C) {
        if (op < 0x1C) { static const uint8_t k[7] = {ADD, SUB, MADD, MSUB, MAX, MIN, MUL}; A(k[op >> 2], 1, false); return d; }
        switch (op) {
        case 0x1C: A(MUL, 2, false); return d;  case 0x1D: A(MAX, 3, false); return d;
        case 0x1E: A(MUL, 3, false); return d;  case 0x1F: A(MIN, 3, false); return d;
        case 0x20: A(ADD, 2, false); return d;  case 0x21: A(MADD, 2, false); return d;
        case 0x22: A(ADD, 3, false); return d;  case 0x23: A(MADD, 3, false); return d;
        case 0x24: A(SUB, 2, false); return d;  case 0x25: A(MSUB, 2, false); return d;
        case 0x26: A(SUB, 3, false); return d;  case 0x27: A(MSUB, 3, false); return d;
        case 0x28: A(ADD, 0, false); return d;  case 0x29: A(MADD, 0, false); return d;
        case 0x2A: A(MUL, 0, false); return d;  case 0x2B: A(MAX, 0, false); return d;
        case 0x2C: A(SUB, 0, false); return d;  case 0x2D: A(MSUB, 0, false); return d;
        case 0x2F: A(MIN, 0, false); return d;
        default: d.kind = UK_SLOW; return d;
        }
    }
    int sop = ((w >> 6) & 31) * 4 + (w & 3);
    if (sop < 0x10) { static const uint8_t k[4] = {ADD, SUB, MADD, MSUB}; A(k[sop >> 2], 1, true); return d; }
    if (sop < 0x14) { d.kind = UK_ITOF; d.sh = (uint8_t)(sop & 3); return d; }
    if (sop < 0x18) { d.kind = UK_FTOI; d.sh = (uint8_t)(sop & 3); return d; }
    switch (sop) {
    case 0x18: case 0x19: case 0x1A: case 0x1B: A(MUL, 1, true); return d;
    case 0x1C: A(MUL, 2, true); return d;   case 0x1D: d.kind = UK_ABS; return d;
    case 0x1E: A(MUL, 3, true); return d;
    case 0x20: A(ADD, 2, true); return d;   case 0x21: A(MADD, 2, true); return d;
    case 0x22: A(ADD, 3, true); return d;   case 0x23: A(MADD, 3, true); return d;
    case 0x24: A(SUB, 2, true); return d;   case 0x25: A(MSUB, 2, true); return d;
    case 0x26: A(SUB, 3, true); return d;   case 0x27: A(MSUB, 3, true); return d;
    case 0x28: A(ADD, 0, true); return d;   case 0x29: A(MADD, 0, true); return d;
    case 0x2A: A(MUL, 0, true); return d;   case 0x2C: A(SUB, 0, true); return d;
    case 0x2D: A(MSUB, 0, true); return d;
    case 0x2F: d.kind = UK_NOP; return d;
    default: d.kind = UK_SLOW; return d;   // CLIP, OPMULA, OPMSUB, unknown
    }
}

static inline __m128 vclamp4(__m128 v) {
    const __m128 signm = _mm_castsi128_ps(_mm_set1_epi32((int)0x80000000));
    __m128 sign = _mm_and_ps(v, signm), a = _mm_andnot_ps(signm, v);
    __m128 big = _mm_cmpnle_ps(a, _mm_set1_ps(FLT_MAX));   // > FLT_MAX or NaN
    __m128 den = _mm_and_ps(_mm_cmplt_ps(a, _mm_set1_ps(FLT_MIN)), _mm_cmpneq_ps(a, _mm_setzero_ps()));
    v = _mm_or_ps(_mm_andnot_ps(big, v), _mm_and_ps(big, _mm_or_ps(sign, _mm_set1_ps(FLT_MAX))));
    return _mm_or_ps(_mm_andnot_ps(den, v), _mm_and_ps(den, sign));
}
static inline __m128i sm_key4(__m128 v) {
    __m128i i = _mm_castps_si128(v);
    return _mm_sign_epi32(_mm_and_si128(i, _mm_set1_epi32(0x7FFFFFFF)), i);
}
static const uint8_t k_rev4[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};

static inline void set_mac4(R5900 *r, int dest, __m128 res) {
    int dm = k_rev4[dest];   // movemask lane order (x = bit 0)
    int zm = _mm_movemask_ps(_mm_cmpeq_ps(res, _mm_setzero_ps())) & dm;   // denormals were flushed by vclamp4
    int sm = _mm_movemask_ps(res) & dm & (vu0_flags_legacy ? ~zm : 0xF);   // S = sign bit: -0 sets Z and S
    int om = _mm_movemask_ps(_mm_cmpge_ps(_mm_andnot_ps(_mm_castsi128_ps(_mm_set1_epi32((int)0x80000000)), res),
                                          _mm_set1_ps(FLT_MAX))) & dm;
    uint32_t mac = k_rev4[zm] | (k_rev4[sm] << 4) | (k_rev4[om] << 12);
    uint32_t st = r->vi[VI_STATUS] & 0xFF0;
    if (mac & 0x000F) st |= 0x001 | 0x040;
    if (mac & 0x00F0) st |= 0x002 | 0x080;
    if (mac & 0xF000) st |= 0x008 | 0x200;
    r->vi[VI_MAC] = mac;
    r->vi[VI_STATUS] = (r->vi[VI_STATUS] & 0x030) | st | (r->vi[VI_STATUS] & 0xC00);
}

static __forceinline void upper_fast(VuState &vu, const UDec &d, uint32_t w, UpperResult &out) {
    R5900 *r = vu.regs;
    switch (d.kind) {
    case UK_NOP: return;
    case UK_SLOW: upper(vu, w, out); return;
    case UK_ITOF: {
        static const float sc[4] = {1.0f, 1.0f / 16, 1.0f / 4096, 1.0f / 32768};
        __m128 v = _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)&r->vf[d.fs])), _mm_set1_ps(sc[d.sh]));
        out.kind = 1; out.reg = d.ft; out.dest = d.dest; _mm_storeu_ps(out.v.f, v);
        return;
    }
    case UK_FTOI: {
        static const int sh[4] = {0, 4, 12, 15};
        const R128 &S = r->vf[d.fs];
        out.kind = 1; out.reg = d.ft; out.dest = d.dest;
        for (int k = 0; k < 4; k++) out.v.sw[k] = vu_ftoi(S.f[k], sh[d.sh]);
        return;
    }
    case UK_ABS: {
        __m128 v = _mm_andnot_ps(_mm_castsi128_ps(_mm_set1_epi32((int)0x80000000)), _mm_loadu_ps(r->vf[d.fs].f));
        out.kind = 1; out.reg = d.ft; out.dest = d.dest; _mm_storeu_ps(out.v.f, v);
        return;
    }
    }
    enum { ADD, SUB, MADD, MSUB, MAX, MIN, MUL };
    __m128 a = _mm_loadu_ps(r->vf[d.fs].f), b;
    switch (d.src) {
    case 0: b = _mm_loadu_ps(r->vf[d.ft].f); break;
    case 1: b = _mm_set1_ps(r->vf[d.ft].f[d.bc]); break;
    case 2: b = _mm_set1_ps(r->vq); break;
    default: { float I; memcpy(&I, &r->vi[VI_I], 4); b = _mm_set1_ps(I); break; }
    }
    __m128 v;
    bool flags = true;
    switch (d.op) {
    case ADD: v = vclamp4(_mm_add_ps(a, b)); break;
    case SUB: v = vclamp4(_mm_sub_ps(a, b)); break;
    case MADD: v = vclamp4(_mm_add_ps(_mm_loadu_ps(r->vacc.f), _mm_mul_ps(a, b))); break;
    case MSUB: v = vclamp4(_mm_sub_ps(_mm_loadu_ps(r->vacc.f), _mm_mul_ps(a, b))); break;
    case MAX: { __m128 sel = _mm_castsi128_ps(_mm_cmpgt_epi32(sm_key4(b), sm_key4(a))); v = _mm_or_ps(_mm_and_ps(sel, b), _mm_andnot_ps(sel, a)); flags = false; break; }
    case MIN: { __m128 sel = _mm_castsi128_ps(_mm_cmpgt_epi32(sm_key4(a), sm_key4(b))); v = _mm_or_ps(_mm_and_ps(sel, b), _mm_andnot_ps(sel, a)); flags = false; break; }
    default: v = vclamp4(_mm_mul_ps(a, b)); break;
    }
    out.kind = d.acc ? 2 : 1; out.reg = d.acc ? 0 : d.fd; out.dest = d.dest; out.flags = flags;
    _mm_storeu_ps(out.v.f, v);
}

// Specialised upper handlers (one indirect call per instruction instead of nested switches).
template <int OP, int SRC, bool ACC>
static void up_arith(VuState &vu, const UDec &d, uint32_t, UpperResult &out) {
    R5900 *r = vu.regs;
    enum { ADD, SUB, MADD, MSUB, MAX, MIN, MUL };
    __m128 a = _mm_loadu_ps(r->vf[d.fs].f), b;
    if constexpr (SRC == 0) b = _mm_loadu_ps(r->vf[d.ft].f);
    else if constexpr (SRC == 1) b = _mm_set1_ps(r->vf[d.ft].f[d.bc]);
    else if constexpr (SRC == 2) b = _mm_set1_ps(r->vq);
    else { float I; memcpy(&I, &r->vi[VI_I], 4); b = _mm_set1_ps(I); }
    __m128 v;
    if constexpr (OP == ADD) v = vclamp4(_mm_add_ps(a, b));
    else if constexpr (OP == SUB) v = vclamp4(_mm_sub_ps(a, b));
    else if constexpr (OP == MADD) v = vclamp4(_mm_add_ps(_mm_loadu_ps(r->vacc.f), _mm_mul_ps(a, b)));
    else if constexpr (OP == MSUB) v = vclamp4(_mm_sub_ps(_mm_loadu_ps(r->vacc.f), _mm_mul_ps(a, b)));
    else if constexpr (OP == MAX) { __m128 sel = _mm_castsi128_ps(_mm_cmpgt_epi32(sm_key4(b), sm_key4(a))); v = _mm_or_ps(_mm_and_ps(sel, b), _mm_andnot_ps(sel, a)); }
    else if constexpr (OP == MIN) { __m128 sel = _mm_castsi128_ps(_mm_cmpgt_epi32(sm_key4(a), sm_key4(b))); v = _mm_or_ps(_mm_and_ps(sel, b), _mm_andnot_ps(sel, a)); }
    else v = vclamp4(_mm_mul_ps(a, b));
    out.kind = ACC ? 2 : 1; out.reg = ACC ? 0 : d.fd; out.dest = d.dest; out.flags = OP != MAX && OP != MIN;
    _mm_storeu_ps(out.v.f, v);
}
static void up_generic(VuState &vu, const UDec &d, uint32_t w, UpperResult &out) { upper_fast(vu, d, w, out); }

// Direct variants: write the destination (and MAC/STATUS) immediately. Used when the paired lower
// op has already run and could not observe the difference.
static __forceinline void commit_direct(R5900 *r, R128 *t, int dest, __m128 v, bool flags) {
    if (flags) set_mac4(r, dest, v);
    if (t == &r->vf[0]) return;
    __m128 m = _mm_castsi128_ps(k_dest_mask[dest]);
    _mm_storeu_ps(t->f, _mm_or_ps(_mm_and_ps(m, v), _mm_andnot_ps(m, _mm_loadu_ps(t->f))));
}
template <int OP, int SRC, bool ACC, bool FL = true>
static void upd_arith(VuState &vu, const UDec &d) {
    R5900 *r = vu.regs;
    enum { ADD, SUB, MADD, MSUB, MAX, MIN, MUL };
    __m128 a = _mm_loadu_ps(r->vf[d.fs].f), b;
    if constexpr (SRC == 0) b = _mm_loadu_ps(r->vf[d.ft].f);
    else if constexpr (SRC == 1) b = _mm_set1_ps(r->vf[d.ft].f[d.bc]);
    else if constexpr (SRC == 2) b = _mm_set1_ps(r->vq);
    else { float I; memcpy(&I, &r->vi[VI_I], 4); b = _mm_set1_ps(I); }
    __m128 v;
    if constexpr (OP == ADD) v = vclamp4(_mm_add_ps(a, b));
    else if constexpr (OP == SUB) v = vclamp4(_mm_sub_ps(a, b));
    else if constexpr (OP == MADD) v = vclamp4(_mm_add_ps(_mm_loadu_ps(r->vacc.f), _mm_mul_ps(a, b)));
    else if constexpr (OP == MSUB) v = vclamp4(_mm_sub_ps(_mm_loadu_ps(r->vacc.f), _mm_mul_ps(a, b)));
    else if constexpr (OP == MAX) { __m128 sel = _mm_castsi128_ps(_mm_cmpgt_epi32(sm_key4(b), sm_key4(a))); v = _mm_or_ps(_mm_and_ps(sel, b), _mm_andnot_ps(sel, a)); }
    else if constexpr (OP == MIN) { __m128 sel = _mm_castsi128_ps(_mm_cmpgt_epi32(sm_key4(a), sm_key4(b))); v = _mm_or_ps(_mm_and_ps(sel, b), _mm_andnot_ps(sel, a)); }
    else v = vclamp4(_mm_mul_ps(a, b));
    commit_direct(r, ACC ? &r->vacc : &r->vf[d.fd], d.dest, v, FL && OP != MAX && OP != MIN);
}
static void upd_itof(VuState &vu, const UDec &d) {
    static const float sc[4] = {1.0f, 1.0f / 16, 1.0f / 4096, 1.0f / 32768};
    R5900 *r = vu.regs;
    __m128 v = _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)&r->vf[d.fs])), _mm_set1_ps(sc[d.sh]));
    commit_direct(r, &r->vf[d.ft], d.dest, v, false);
}
static __forceinline __m128 ftoi4(__m128 x, int shc) {
    static const float sc[4] = {1.0f, 16.0f, 4096.0f, 32768.0f};
    x = _mm_mul_ps(x, _mm_set1_ps(sc[shc]));
    __m128i i = _mm_cvttps_epi32(x);                                     // overflow / NaN -> 0x80000000
    __m128 pos = _mm_cmpnlt_ps(x, _mm_set1_ps(2147483648.0f));            // >= 2^31 or NaN -> 0x7FFFFFFF
    return _mm_castsi128_ps(_mm_xor_si128(i, _mm_and_si128(_mm_castps_si128(pos), _mm_set1_epi32(-1))));
}
static void upd_ftoi(VuState &vu, const UDec &d) {
    R5900 *r = vu.regs;
    commit_direct(r, &r->vf[d.ft], d.dest, ftoi4(_mm_loadu_ps(r->vf[d.fs].f), d.sh), false);
}
static void upd_abs(VuState &vu, const UDec &d) {
    R5900 *r = vu.regs;
    __m128 v = _mm_andnot_ps(_mm_castsi128_ps(_mm_set1_epi32((int)0x80000000)), _mm_loadu_ps(r->vf[d.fs].f));
    commit_direct(r, &r->vf[d.ft], d.dest, v, false);
}
static void upd_nop(VuState &, const UDec &) {}
template <int OP, int SRC> static UpperDirectFn dpick_src(bool acc, bool fl) {
    return acc ? (fl ? &upd_arith<OP, SRC, true, true> : &upd_arith<OP, SRC, true, false>)
               : (fl ? &upd_arith<OP, SRC, false, true> : &upd_arith<OP, SRC, false, false>);
}
template <int OP> static UpperDirectFn dpick_op(int src, bool acc, bool fl) {
    switch (src) { case 0: return dpick_src<OP, 0>(acc, fl); case 1: return dpick_src<OP, 1>(acc, fl); case 2: return dpick_src<OP, 2>(acc, fl); default: return dpick_src<OP, 3>(acc, fl); }
}
// fl = false: skip MAC/STATUS updates (only when no code in micro memory can read them)
static UpperDirectFn pick_upper_direct(const UDec &d, bool fl = true) {
    switch (d.kind) {
    case UK_NOP: return &upd_nop;
    case UK_ITOF: return &upd_itof;
    case UK_FTOI: return &upd_ftoi;
    case UK_ABS: return &upd_abs;
    case UK_ARITH:
        switch (d.op) {
        case 0: return dpick_op<0>(d.src, d.acc, fl); case 1: return dpick_op<1>(d.src, d.acc, fl);
        case 2: return dpick_op<2>(d.src, d.acc, fl); case 3: return dpick_op<3>(d.src, d.acc, fl);
        case 4: return dpick_op<4>(d.src, d.acc, fl); case 5: return dpick_op<5>(d.src, d.acc, fl);
        default: return dpick_op<6>(d.src, d.acc, fl);
        }
    default: return nullptr;   // CLIP / OPMULA / OPMSUB / unknown: staged path
    }
}
static void up_nop(VuState &, const UDec &, uint32_t, UpperResult &) {}

template <int OP, int SRC> static UpperFn pick_src(bool acc) { return acc ? &up_arith<OP, SRC, true> : &up_arith<OP, SRC, false>; }
template <int OP> static UpperFn pick_op(int src, bool acc) {
    switch (src) { case 0: return pick_src<OP, 0>(acc); case 1: return pick_src<OP, 1>(acc); case 2: return pick_src<OP, 2>(acc); default: return pick_src<OP, 3>(acc); }
}
static UpperFn pick_upper(const UDec &d) {
    if (d.kind == UK_NOP) return &up_nop;
    if (d.kind != UK_ARITH) return &up_generic;
    switch (d.op) {
    case 0: return pick_op<0>(d.src, d.acc); case 1: return pick_op<1>(d.src, d.acc);
    case 2: return pick_op<2>(d.src, d.acc); case 3: return pick_op<3>(d.src, d.acc);
    case 4: return pick_op<4>(d.src, d.acc); case 5: return pick_op<5>(d.src, d.acc);
    default: return pick_op<6>(d.src, d.acc);
    }
}

static __forceinline void commit_upper_fast(VuState &vu, const UpperResult &u) {
    R5900 *r = vu.regs;
    if (u.clip >= 0) r->vi[VI_CLIP] = (uint32_t)u.clip;
    if (!u.kind) return;
    __m128 v = _mm_loadu_ps(u.v.f);
    if (u.flags) set_mac4(r, u.dest, v);
    if (u.kind == 1 && u.reg == 0) return;
    R128 *t = u.kind == 2 ? &r->vacc : &r->vf[u.reg];
    static const __m128i masks[16] = {
#define M(d) _mm_setr_epi32((d & 8) ? -1 : 0, (d & 4) ? -1 : 0, (d & 2) ? -1 : 0, (d & 1) ? -1 : 0)
        M(0), M(1), M(2), M(3), M(4), M(5), M(6), M(7), M(8), M(9), M(10), M(11), M(12), M(13), M(14), M(15)
#undef M
    };
    __m128 m = _mm_castsi128_ps(masks[u.dest]);
    _mm_storeu_ps(t->f, _mm_or_ps(_mm_and_ps(m, v), _mm_andnot_ps(m, _mm_loadu_ps(t->f))));
}

#include <chrono>
extern uint64_t g_prof_vu_ns;
static void vu_execute_impl(VuState &vu, uint32_t start_pc);
uint32_t g_vu1_last_start;
uint64_t g_vu_instr;

static void decode_pair(VuDec &dc, uint32_t lo, uint32_t up) {
    dc.lo = lo; dc.up = up; dc.valid = true;
    flatten_hazards(dc, decode_hazards(up, lo));
    // MOVE with an empty dest / to vf0 does nothing (the usual lower NOP)
    dc.lower_nop = (up & 0x80000000u) || ((lo >> 25) == 0x40 && (lo & 0x7FF) == 0x33C && (((lo >> 21) & 15) == 0 || ((lo >> 16) & 31) == 0));
    dc.u = decode_upper(up);
    dc.u.fn = pick_upper(dc.u);
    dc.l = decode_lower(lo);
    dc.ud = pick_upper_direct(dc.u);
    if (dc.ud && !dc.lower_nop) {
        int w = dc.l.vfw;
        bool reads_ft = dc.u.kind == UK_ARITH && (dc.u.src <= 1);
        if (w < 0 || (w != 0 && (w == dc.u.fs || (reads_ft && w == dc.u.ft)))) dc.ud = nullptr;
    }
}

// ================================================================ VU1 call-threaded JIT
// Straight-line runs of microcode are compiled into x64 sequences of direct calls to the
// pre-decoded handlers above (one call per pipeline step / lower op / upper op), so every
// dispatch is a predictable direct call instead of two data-dependent indirect ones.
// Semantics are exactly the interpreter's (vubench checksum); odd cases (branch in a delay
// slot, E-bit next to a branch) fall back to the interpreter for the rest of the program.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <vector>

static uint32_t g_jit_br;
static int g_jit_taken;
static uint32_t g_micro_gen[2] = {1, 1};
void vu_micro_dirty(int index) { g_micro_gen[index & 1]++; }
// Does anything in micro memory read the MAC / STATUS flags (FSxxx / FMxxx)? If not, compiled
// code skips computing them.
// Returns 0 (nothing reads flags), 1 (only MAC is read: FMxxx) or 2 (STATUS is read: FSxxx).
static uint32_t g_flags_gen[2];
static int g_flags_needed[2] = {2, 2};
static bool reads_mac(const VuDec &d) { uint32_t op = d.lo >> 25; return !d.lower_nop && op >= 0x18 && op <= 0x1B; }
static int flags_needed(VuState &vu) {
    int idx = vu.index & 1;
    if (g_flags_gen[idx] != g_micro_gen[idx]) {
        g_flags_gen[idx] = g_micro_gen[idx];
        int need = 0;
        for (uint32_t pc = 0; pc <= vu.micro_mask && need < 2; pc += 8) {
            uint32_t lo, up; memcpy(&lo, vu.micro + pc, 4); memcpy(&up, vu.micro + pc + 4, 4);
            uint32_t op = lo >> 25;
            if (up & 0x80000000u) continue;
            if (op >= 0x14 && op <= 0x17) need = 2;
            else if (op >= 0x18 && op <= 0x1B) need = 1;
        }
        g_flags_needed[idx] = need;
    }
    return g_flags_needed[idx];
}

template <bool RD, bool WR, bool WQP, bool IBIT>
static void jpre(VuState &vu, const VuDec &dc) {
    vu.cycle++;
    R5900 *r = vu.regs;
    if (vu.q_ready && vu.cycle >= vu.q_ready) { r->vq = vu.q_pending; vu.q_ready = 0; }
    if (vu.p_ready && vu.cycle >= vu.p_ready) { vu.p = vu.p_pending; vu.p_ready = 0; }
    if constexpr (IBIT) r->vi[VI_I] = dc.lo;
    int64_t *ready = &vu.vf_ready[0][0];
    if constexpr (RD || WQP) {
        int64_t need = vu.cycle;
        if constexpr (RD) for (int i = 0; i < dc.nrd; i++) need = std::max(need, ready[dc.rd[i]]);
        if constexpr (WQP) {
            if (dc.wait_q && vu.q_ready) need = std::max(need, vu.q_ready);
            if (dc.wait_p && vu.p_ready) need = std::max(need, vu.p_ready);
        }
        if (need > vu.cycle) { vu.cycle = need; commit_q(vu, false); commit_p(vu, false); }
    }
    if constexpr (WR) for (int i = 0; i < dc.nwr; i++) ready[dc.wr[i]] = vu.cycle + 4;
}
typedef void (*PreFn)(VuState &, const VuDec &);
static PreFn pick_pre(const VuDec &dc, bool ibit) {
    static const PreFn t[16] = {
        &jpre<0, 0, 0, 0>, &jpre<0, 0, 0, 1>, &jpre<0, 0, 1, 0>, &jpre<0, 0, 1, 1>,
        &jpre<0, 1, 0, 0>, &jpre<0, 1, 0, 1>, &jpre<0, 1, 1, 0>, &jpre<0, 1, 1, 1>,
        &jpre<1, 0, 0, 0>, &jpre<1, 0, 0, 1>, &jpre<1, 0, 1, 0>, &jpre<1, 0, 1, 1>,
        &jpre<1, 1, 0, 0>, &jpre<1, 1, 0, 1>, &jpre<1, 1, 1, 0>, &jpre<1, 1, 1, 1>};
    int k = (dc.nrd ? 8 : 0) | (dc.nwr ? 4 : 0) | ((dc.wait_q || dc.wait_p) ? 2 : 0) | (ibit ? 1 : 0);
    return t[k];
}
static int jl_generic(VuState &vu, const LDec &, uint32_t lo, uint32_t pc, uint32_t *br) {
    int t = lower(vu, lo, pc, br);
    vu.regs->vf[0] = VF0; vu.regs->vi[0] = 0;
    return t;
}
static int jstaged(VuState &vu, const VuDec &dc, uint32_t pc) {
    UpperResult u;
    dc.u.fn(vu, dc.u, dc.up, u);
    int taken = dc.lower_nop ? 0 : dc.l.fn(vu, dc.l, dc.lo, pc, &g_jit_br);
    commit_upper_fast(vu, u);
    vu.regs->vf[0] = VF0; vu.regs->vi[0] = 0;
    return taken;
}
static uint32_t jend_branch(uint32_t fall) { return g_jit_taken ? g_jit_br : fall; }
static uint32_t jend_e(VuState &vu, uint32_t pc) { vu.tpc = pc; vu.regs->vi[VI_TPC] = pc / 8; return 0xFFFFFFFFu; }

struct JitBlock {
    uint32_t gen = 0, n = 0;            // n = pairs executed (for the instruction counter)
    bool interp = false;
    int flags = 2;
    std::vector<uint32_t> words;        // micro words the block was compiled from (lo, up per pair)
    std::vector<VuDec> dec;
    uint32_t (*fn)(VuState *) = nullptr;
};
static JitBlock *g_blk[2][2048];

static uint8_t *g_code, *g_code_end, *g_code_ptr;
struct Emit {
    uint8_t *p;
    void b(uint8_t x) { *p++ = x; }
    void d(uint32_t x) { memcpy(p, &x, 4); p += 4; }
    void q(uint64_t x) { memcpy(p, &x, 8); p += 8; }
    void call(const void *fn) { b(0x48); b(0xB8); q((uint64_t)fn); b(0xFF); b(0xD0); }   // mov rax, fn; call rax
    void rcx_vu() { b(0x48); b(0x89); b(0xD9); }                                        // mov rcx, rbx
    void rdx(const void *v) { b(0x48); b(0xBA); q((uint64_t)v); }                       // mov rdx, imm64
    void edx(uint32_t v) { b(0xBA); d(v); }
    void ecx(uint32_t v) { b(0xB9); d(v); }
    void r8d(uint32_t v) { b(0x41); b(0xB8); d(v); }
    void r9d(uint32_t v) { b(0x41); b(0xB9); d(v); }
    void arg5(const void *v) { b(0x48); b(0xB8); q((uint64_t)v); b(0x48); b(0x89); b(0x44); b(0x24); b(0x20); }  // mov [rsp+32], rax
    void store_taken() { b(0xA3); q((uint64_t)&g_jit_taken); }                         // mov [g_jit_taken], eax
    void epilogue() { b(0x48); b(0x83); b(0xC4); b(0x28); b(0x5E); b(0x5B); b(0xC3); } // add rsp,40; pop rsi; pop rbx; ret
    // SSE (legacy encoding, xmm0-7 only). Memory operands: [rsi + disp32] (the R5900 register
    // file) or [rax + disp8] (constants; must be 16-byte aligned for the arithmetic forms).
    void rm_rsi(int x, uint32_t disp) { b((uint8_t)(0x80 | (x << 3) | 6)); d(disp); }
    void rm_rax(int x, uint8_t disp) { b((uint8_t)(0x40 | (x << 3))); b(disp); }
    void rr(int dst, int src) { b((uint8_t)(0xC0 | (dst << 3) | src)); }
    void movups_ld(int x, uint32_t off) { b(0x0F); b(0x10); rm_rsi(x, off); }
    void movups_st(uint32_t off, int x) { b(0x0F); b(0x11); rm_rsi(x, off); }
    void movss_ld(int x, uint32_t off) { b(0xF3); b(0x0F); b(0x10); rm_rsi(x, off); }
    void bcast(int x) { b(0x0F); b(0xC6); rr(x, x); b(0x00); }                      // shufps x, x, 0
    void ps(uint8_t op, int dst, int src) { b(0x0F); b(op); rr(dst, src); }         // addps 58 mulps 59 subps 5C andps 54 andnps 55 orps 56 xorps 57 movaps 28
    void ps_rax(uint8_t op, int dst, uint8_t disp) { b(0x0F); b(op); rm_rax(dst, disp); }
    void cmpps_rax(int dst, uint8_t disp, uint8_t pred) { b(0x0F); b(0xC2); rm_rax(dst, disp); b(pred); }
    void cmpps(int dst, int src, uint8_t pred) { b(0x0F); b(0xC2); rr(dst, src); b(pred); }
    void rax(const void *v) { b(0x48); b(0xB8); q((uint64_t)v); }
    void rcx_regs() { b(0x48); b(0x89); b(0xF1); }                                    // mov rcx, rsi
    void eax_vi(int n) { b(0x8B); b(0x86); d((uint32_t)(offsetof(R5900, vi) + 4 * n)); }   // mov eax, [rsi + vi[n]]
    void movzx_eax_ax() { b(0x0F); b(0xB7); b(0xC0); }
    void ecx_eax() { b(0x89); b(0xC1); }
    void shl_eax4() { b(0xC1); b(0xE0); b(0x04); }
    void and_eax_rbx(uint32_t off) { b(0x23); b(0x83); d(off); }                       // and eax, [rbx + off]
    void add_rax_rbx(uint32_t off) { b(0x48); b(0x03); b(0x83); d(off); }              // add rax, [rbx + off]
    void movups_ld_rax(int x) { b(0x0F); b(0x10); b((uint8_t)(x << 3)); }             // movups x, [rax]
    void movups_st_rax(int x) { b(0x0F); b(0x11); b((uint8_t)(x << 3)); }             // movups [rax], x
    void inc_ecx_mask() { b(0xFF); b(0xC1); b(0x81); b(0xE1); d(0xFFFF); }            // inc ecx; and ecx, 0xFFFF
    void vi_ecx(int n) { b(0x89); b(0x8E); d((uint32_t)(offsetof(R5900, vi) + 4 * n)); }  // mov [rsi + vi[n]], ecx
    void rdx_rax() { b(0x48); b(0x89); b(0xC2); }                                     // mov rdx, rax
    void movaps_ld_rdx(int x) { b(0x0F); b(0x10); b((uint8_t)(0x02 | (x << 3))); }    // movups x, [rdx]
    void movups_st_rdx(int x) { b(0x0F); b(0x11); b((uint8_t)(0x02 | (x << 3))); }    // movups [rdx], x
};
enum { SSE_ADD = 0x58, SSE_MUL = 0x59, SSE_SUB = 0x5C, SSE_AND = 0x54, SSE_ANDN = 0x55, SSE_OR = 0x56, SSE_XOR = 0x57, SSE_MOV = 0x28 };
alignas(16) static const uint32_t k_jit_const[12] = {
    0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u,   // sign mask
    0x7F7FFFFFu, 0x7F7FFFFFu, 0x7F7FFFFFu, 0x7F7FFFFFu,   // FLT_MAX
    0x00800000u, 0x00800000u, 0x00800000u, 0x00800000u};  // FLT_MIN

alignas(16) static float g_jit_res[4];
static void jit_setmac(R5900 *r, int dest) { set_mac4(r, dest, _mm_load_ps(g_jit_res)); }
static void jit_qp(VuState &vu) {   // jpre's Q/P commit (cycle already advanced)
    R5900 *r = vu.regs;
    if (vu.q_ready && vu.cycle >= vu.q_ready) { r->vq = vu.q_pending; vu.q_ready = 0; }
    if (vu.p_ready && vu.cycle >= vu.p_ready) { vu.p = vu.p_pending; vu.p_ready = 0; }
}
// xmm<val> blended into vf[reg] under dest (clobbers xmm0/xmm1 unless val is one of them)
static void emit_put_vf(Emit &e, int reg, int dest, int val) {
    const uint32_t off = (uint32_t)offsetof(R5900, vf) + 16u * reg;
    if (!reg) return;
    if (dest == 15) { e.movups_st(off, val); return; }
    int m = val == 1 ? 2 : 1, o = val == 0 ? 2 : 0;
    e.rax(&k_dest_mask[dest]); e.ps_rax(SSE_MOV, m, 0);
    e.movups_ld(o, off);
    e.ps(SSE_AND, val, m); e.ps(SSE_ANDN, m, o); e.ps(SSE_OR, m, val);
    e.movups_st(off, m);
}
// Inline lower ops: MOVE, LQI, SQI (exactly l_move / l_lqi / l_sqi)
static bool emit_lower_inline(Emit &e, const LDec &l) {
    const uint32_t VF = (uint32_t)offsetof(R5900, vf);
    if (l.fn == &l_move) {
        e.movups_ld(3, VF + 16u * l.is);
        emit_put_vf(e, l.it, l.dest, 3);
        return true;
    }
    if (l.fn == &l_lqi || l.fn == &l_sqi) {
        const bool load = l.fn == &l_lqi;
        const int ireg = load ? l.is : l.it;
        e.eax_vi(ireg); e.movzx_eax_ax(); e.ecx_eax();                 // a = VI & 0xFFFF
        e.shl_eax4(); e.and_eax_rbx((uint32_t)offsetof(VuState, mem_mask));
        e.add_rax_rbx((uint32_t)offsetof(VuState, mem));              // rax = qptr(a)
        if (load) {
            e.movups_ld_rax(3);
            emit_put_vf(e, l.it, l.dest, 3);
        } else {
            e.movups_ld(3, VF + 16u * l.is);
            if (l.dest == 15) e.movups_st_rax(3);
            else {
                e.rdx_rax();
                e.rax(&k_dest_mask[l.dest]); e.ps_rax(SSE_MOV, 1, 0);
                e.movaps_ld_rdx(0);
                e.ps(SSE_AND, 3, 1); e.ps(SSE_ANDN, 1, 0); e.ps(SSE_OR, 1, 3);
                e.movups_st_rdx(1);
            }
        }
        if (ireg) { e.inc_ecx_mask(); e.vi_ecx(ireg); }
        return true;
    }
    return false;
}
// jpre<0,0,0,0> inline: cycle++, then the Q/P commit only if one is pending
static void emit_pre_plain(Emit &e) {
    e.b(0x48); e.b(0xFF); e.b(0x83); e.d((uint32_t)offsetof(VuState, cycle));          // inc qword [rbx + cycle]
    e.b(0x48); e.b(0x8B); e.b(0x83); e.d((uint32_t)offsetof(VuState, q_ready));        // mov rax, [rbx + q_ready]
    e.b(0x48); e.b(0x0B); e.b(0x83); e.d((uint32_t)offsetof(VuState, p_ready));        // or rax, [rbx + p_ready]
    e.b(0x74); e.b(15);                                                                 // jz +15
    e.rcx_vu(); e.call((const void *)&jit_qp);                                          // 3 + 12 bytes
}
// Inline FMAC (ADD/SUB/MUL/MADD/MSUB, any source, flags dead): the same instruction sequence as
// upd_arith + vclamp4 + commit_direct, so results are bit-identical.
static bool emit_upper_inline(Emit &e, const UDec &u, bool flags) {
    if (u.kind != UK_ARITH || u.op == 4 || u.op == 5) return false;
    const uint32_t VF = (uint32_t)offsetof(R5900, vf), ACCO = (uint32_t)offsetof(R5900, vacc);
    const uint32_t tgt = u.acc ? ACCO : VF + 16u * u.fd;
    const bool novf = !u.acc && u.fd == 0;           // writes vf0: only the flags (if any) remain
    if (novf && !flags) return true;
    e.movups_ld(0, VF + 16u * u.fs);                 // xmm0 = a
    switch (u.src) {                                 // xmm1 = b
    case 0: e.movups_ld(1, VF + 16u * u.ft); break;
    case 1: e.movss_ld(1, VF + 16u * u.ft + 4u * u.bc); e.bcast(1); break;
    case 2: e.movss_ld(1, (uint32_t)offsetof(R5900, vq)); e.bcast(1); break;
    default: e.movss_ld(1, (uint32_t)offsetof(R5900, vi) + 4u * VI_I); e.bcast(1); break;
    }
    enum { ADD, SUB, MADD, MSUB, MAX, MIN, MUL };
    switch (u.op) {                                  // xmm0 = unclamped result
    case ADD: e.ps(SSE_ADD, 0, 1); break;
    case SUB: e.ps(SSE_SUB, 0, 1); break;
    case MUL: e.ps(SSE_MUL, 0, 1); break;
    case MADD: e.ps(SSE_MUL, 0, 1); e.movups_ld(1, ACCO); e.ps(SSE_ADD, 1, 0); e.ps(SSE_MOV, 0, 1); break;
    default: e.ps(SSE_MUL, 0, 1); e.movups_ld(1, ACCO); e.ps(SSE_SUB, 1, 0); e.ps(SSE_MOV, 0, 1); break;   // MSUB
    }
    // vclamp4: xmm2 = sign, xmm3 = |v|, xmm4 = big, v' = big ? sign|FLT_MAX : v; den -> sign
    e.rax(k_jit_const);
    e.ps(SSE_MOV, 2, 0); e.ps_rax(SSE_AND, 2, 0);
    e.ps_rax(SSE_MOV, 3, 0); e.ps(SSE_ANDN, 3, 0);
    e.ps(SSE_MOV, 4, 3); e.cmpps_rax(4, 16, 6);                     // cmpnleps |v|, FLT_MAX
    e.ps(SSE_MOV, 5, 2); e.ps_rax(SSE_OR, 5, 16); e.ps(SSE_AND, 5, 4);
    e.ps(SSE_ANDN, 4, 0); e.ps(SSE_OR, 4, 5);                       // xmm4 = v'
    e.ps(SSE_MOV, 5, 3); e.cmpps_rax(5, 32, 1);                     // cmpltps |v|, FLT_MIN
    e.ps(SSE_XOR, 1, 1); e.cmpps(3, 1, 4); e.ps(SSE_AND, 5, 3);     // & (|v| != 0) -> den
    e.ps(SSE_AND, 2, 5); e.ps(SSE_ANDN, 5, 4); e.ps(SSE_OR, 5, 2);  // xmm5 = result
    if (flags) {   // MAC/STATUS from the full result (helper), then the masked store
        e.rax(g_jit_res); e.b(0x0F); e.b(0x29); e.b(0x28);                // movaps [rax], xmm5
        e.rcx_regs(); e.edx((uint32_t)u.dest); e.call((const void *)&jit_setmac);
        if (novf) return true;
        e.rax(g_jit_res); e.b(0x0F); e.b(0x28); e.b(0x28);                // movaps xmm5, [rax]
    }
    if (u.dest == 15) { e.movups_st(tgt, 5); return true; }
    e.rax(&k_dest_mask[u.dest]);
    e.ps_rax(SSE_MOV, 1, 0);                                        // movaps xmm1, mask
    e.movups_ld(0, tgt);
    e.ps(SSE_AND, 5, 1); e.ps(SSE_ANDN, 1, 0); e.ps(SSE_OR, 1, 5);
    e.movups_st(tgt, 1);
    return true;
}

static bool is_branch(const VuDec &dc) {
    if (dc.lower_nop) return false;
    uint32_t op = dc.lo >> 25;
    return op == 0x20 || op == 0x21 || op == 0x24 || op == 0x25 || op == 0x28 || op == 0x29 || (op >= 0x2C && op <= 0x2F);
}

static void jit_reset() {
    for (auto &v : g_blk) for (JitBlock *&b : v) { delete b; b = nullptr; }
    g_code_ptr = g_code;
}

static JitBlock *jit_compile(VuState &vu, uint32_t start) {
    const uint32_t mask = vu.micro_mask;
    auto words_at = [&](uint32_t pc, uint32_t &lo, uint32_t &up) { memcpy(&lo, vu.micro + pc, 4); memcpy(&up, vu.micro + pc + 4, 4); };
    JitBlock *b = new JitBlock;
    b->gen = g_micro_gen[vu.index];
    b->flags = flags_needed(vu);
    // scan
    enum { END_PLAIN, END_BRANCH, END_E };
    int end = END_PLAIN;
    uint32_t pc = start, next = 0;
    std::vector<VuDec> dec;
    for (int i = 0; i < 256; i++) {
        VuDec dc{}; uint32_t lo, up; words_at(pc, lo, up); decode_pair(dc, lo, up);
        bool br = is_branch(dc), ebit = (up & 0x40000000u) != 0;
        if (br || ebit) {
            uint32_t pc2 = (pc + 8) & mask;
            VuDec d2{}; uint32_t lo2, up2; words_at(pc2, lo2, up2); decode_pair(d2, lo2, up2);
            if ((br && ebit) || is_branch(d2) || (up2 & 0x40000000u)) {   // unusual: leave to the interpreter
                if (dec.empty()) { b->interp = true; return b; }
                next = pc; break;
            }
            dec.push_back(dc); dec.push_back(d2);
            end = br ? END_BRANCH : END_E;
            next = (pc2 + 8) & mask;
            break;
        }
        dec.push_back(dc);
        pc = (pc + 8) & mask;
        next = pc;
    }
    // Static hazard pruning. Every pair takes at least one cycle and FMAC results are usable 4
    // cycles after issue, so a read can only stall on a write from the previous 3 pairs. Past the
    // first 3 pairs of a block those writers are known: check only fields written by them, and
    // record a write only if a reader within the next 3 pairs (or the next block) can look at it.
    // Skipped records leave older, smaller ready times behind, which can never cause a stall.
    {
        const int n = (int)dec.size();
        std::vector<VuDec> full = dec;
        auto writes = [&](int j, uint8_t f) { for (int k = 0; k < full[j].nwr; k++) if (full[j].wr[k] == f) return true; return false; };
        auto reads = [&](int j, uint8_t f) { for (int k = 0; k < full[j].nrd; k++) if (full[j].rd[k] == f) return true; return false; };
        for (int i = 0; i < n; i++) {
            VuDec &d = dec[i];
            if (i >= 3) {
                d.nrd = 0;
                for (int k = 0; k < full[i].nrd; k++) {
                    uint8_t f = full[i].rd[k];
                    if (writes(i - 1, f) || writes(i - 2, f) || writes(i - 3, f)) d.rd[d.nrd++] = f;
                }
            }
            if (i < n - 3) {
                d.nwr = 0;
                for (int k = 0; k < full[i].nwr; k++) {
                    uint8_t f = full[i].wr[k];
                    if (reads(i + 1, f) || reads(i + 2, f) || reads(i + 3, f)) d.wr[d.nwr++] = f;
                }
            }
        }
    }
    if (b->flags == 0) {
        for (auto &d : dec) if (d.ud) d.ud = pick_upper_direct(d.u, false);
    } else if (b->flags == 1) {
        // only the MAC value matters: a direct FMAC's flags are dead if a later direct FMAC in the
        // block overwrites MAC before any FMxxx reads it (a pair's lower runs before its upper)
        auto sets_mac = [](const VuDec &d) { return d.ud && d.u.kind == UK_ARITH && d.u.op != 4 && d.u.op != 5; };
        for (size_t i = 0; i < dec.size(); i++) {
            if (!sets_mac(dec[i])) continue;
            for (size_t k = i + 1; k < dec.size(); k++) {
                if (reads_mac(dec[k])) break;
                if (sets_mac(dec[k])) { dec[i].ud = pick_upper_direct(dec[i].u, false); break; }
            }
        }
    }
    b->dec = std::move(dec);
    b->n = (uint32_t)b->dec.size();
    for (auto &dc : b->dec) { b->words.push_back(dc.lo); b->words.push_back(dc.up); }
    // emit
    if (g_code_end - g_code_ptr < (ptrdiff_t)(b->n * 320 + 256)) { delete b; return nullptr; }
    Emit e{g_code_ptr};
    b->fn = (uint32_t (*)(VuState *))e.p;
    e.b(0x53); e.b(0x56);                           // push rbx; push rsi
    e.b(0x48); e.b(0x83); e.b(0xEC); e.b(0x28);     // sub rsp, 40
    e.b(0x48); e.b(0x89); e.b(0xCB);                // mov rbx, rcx
    e.b(0x48); e.b(0x8B); e.b(0xB3); e.d((uint32_t)offsetof(VuState, regs));   // mov rsi, [rbx + regs]
    pc = start;
    static const bool inl = getenv("NPPL_VUINLINE") == nullptr || atoi(getenv("NPPL_VUINLINE")) != 0;
    for (uint32_t i = 0; i < b->n; i++, pc = (pc + 8) & mask) {
        const VuDec &dc = b->dec[i];
        bool br = is_branch(dc);
        {
            PreFn pf = pick_pre(dc, (dc.up & 0x80000000u) != 0);
            if (inl && pf == &jpre<0, 0, 0, 0>) emit_pre_plain(e);
            else { e.rcx_vu(); e.rdx(&dc); e.call((const void *)pf); }
        }
        if (dc.ud) {
            if (!dc.lower_nop) {
                LowerFn lf = dc.l.fn == &l_generic ? &jl_generic : dc.l.fn;
                if (!(inl && !br && emit_lower_inline(e, dc.l))) {
                    e.rcx_vu(); e.rdx(&dc.l); e.r8d(dc.lo); e.r9d(pc); e.arg5(&g_jit_br); e.call((const void *)lf);
                    if (br) e.store_taken();
                }
            }
            if (dc.ud != &upd_nop && !(inl && dc.u.kind == UK_ARITH &&
                                       (dc.ud == pick_upper_direct(dc.u, false) || dc.ud == pick_upper_direct(dc.u, true)) &&
                                       emit_upper_inline(e, dc.u, dc.ud != pick_upper_direct(dc.u, false)))) {
                e.rcx_vu(); e.rdx(&dc.u); e.call((const void *)dc.ud);
            }
        } else {
            e.rcx_vu(); e.rdx(&dc); e.r8d(pc); e.call((const void *)&jstaged);
            if (br) e.store_taken();
        }
    }
    if (end == END_BRANCH) { e.ecx(next); e.call((const void *)&jend_branch); }
    else if (end == END_E) { e.rcx_vu(); e.edx(next); e.call((const void *)&jend_e); }
    else { e.b(0xB8); e.d(next); }                  // mov eax, next
    e.epilogue();
    g_code_ptr = (uint8_t *)(((uintptr_t)e.p + 15) & ~(uintptr_t)15);
    return b;
}

static void vu_execute_jit(VuState &vu, uint32_t pc) {
    if (!g_code) {
        size_t sz = 64u << 20;
        g_code = (uint8_t *)VirtualAlloc(nullptr, sz, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
        g_code_ptr = g_code; g_code_end = g_code + sz;
    }
    pc &= vu.micro_mask;
    const int idx = vu.index & 1;
    for (uint64_t n = 0; n < 4000000;) {
        uint32_t slot = (pc >> 3) & 2047;
        JitBlock *b = g_blk[idx][slot];
        if (b && b->gen != g_micro_gen[idx]) {
            bool same = false;
            if (!b->interp) {
                uint32_t p2 = pc;
                same = true;
                for (uint32_t i = 0; i < b->n && same; i++, p2 = (p2 + 8) & vu.micro_mask)
                    same = memcmp(&b->words[i * 2], vu.micro + p2, 8) == 0;
            }
            if (same && !b->interp && b->flags == flags_needed(vu)) b->gen = g_micro_gen[idx];
            else { delete b; b = g_blk[idx][slot] = nullptr; }
        }
        if (!b) {
            b = jit_compile(vu, pc);
            if (!b) { jit_reset(); b = jit_compile(vu, pc); }
            g_blk[idx][slot] = b;
        }
        if (b->interp) { vu_execute_impl(vu, pc); return; }
        g_vu_instr += b->n; n += b->n;
        uint32_t next = b->fn(&vu);
        if (next == 0xFFFFFFFFu) return;
        pc = next;
    }
    rt_log(LOG_WARN, "VU%d: program did not finish (runaway?)", vu.index);
}


// ---------------------------------------------------------------- reference interpreter (debug)
// The plain generic upper()/lower() path with the same pipeline model. NPPL_VUCHECK=1 runs every
// VU0 program through both this and the fast interpreter and logs any difference.
static void vu_execute_ref(VuState &vu, uint32_t start_pc) {
    R5900 *r = vu.regs;
    uint32_t pc = start_pc & vu.micro_mask;
    bool branch_pending = false, ending = false;
    uint32_t branch_target = 0;
    for (uint64_t n = 0; n < 4000000; n++) {
        vu.cycle++;
        if (vu.q_ready && vu.cycle >= vu.q_ready) { r->vq = vu.q_pending; vu.q_ready = 0; }
        if (vu.p_ready && vu.cycle >= vu.p_ready) { vu.p = vu.p_pending; vu.p_ready = 0; }
        uint32_t lo, up;
        memcpy(&lo, vu.micro + pc, 4); memcpy(&up, vu.micro + pc + 4, 4);
        if (up & 0x80000000u) r->vi[VI_I] = lo;
        VuDec dc{}; flatten_hazards(dc, decode_hazards(up, lo));
        int64_t *ready = &vu.vf_ready[0][0];
        int64_t need = vu.cycle;
        for (int i = 0; i < dc.nrd; i++) need = std::max(need, ready[dc.rd[i]]);
        if (dc.wait_q && vu.q_ready) need = std::max(need, vu.q_ready);
        if (dc.wait_p && vu.p_ready) need = std::max(need, vu.p_ready);
        if (need > vu.cycle) { vu.cycle = need; commit_q(vu, false); commit_p(vu, false); }
        for (int i = 0; i < dc.nwr; i++) ready[dc.wr[i]] = vu.cycle + 4;
        UpperResult u;
        upper(vu, up, u);
        uint32_t br = 0;
        int taken = (up & 0x80000000u) ? 0 : lower(vu, lo, pc, &br);
        commit_upper_fast(vu, u);
        r->vf[0] = VF0; r->vi[0] = 0;
        bool was_branch = branch_pending, was_ending = ending;
        uint32_t target = branch_target;
        if (up & 0x40000000u) ending = true;
        if (taken) { branch_pending = true; branch_target = br; }
        else if (was_branch) branch_pending = false;
        pc = (pc + 8) & vu.micro_mask;
        if (was_branch) { pc = target; if (!taken) branch_pending = false; }
        if (was_ending) { vu.tpc = pc; r->vi[VI_TPC] = pc / 8; return; }
    }
}
static void vu_check_run(VuState &vu, uint32_t start_pc) {
    static uint64_t runs, bad;
    VuState s0 = vu; R5900 r0 = *vu.regs;
    std::vector<uint8_t> m0(vu.mem, vu.mem + vu.mem_mask + 1);
    vu_execute_ref(vu, start_pc);
    VuState s1 = vu; R5900 r1 = *vu.regs;
    std::vector<uint8_t> m1(vu.mem, vu.mem + vu.mem_mask + 1);
    vu = s0; *vu.regs = r0; memcpy(vu.mem, m0.data(), m0.size());
    vu_execute_impl(vu, start_pc);
    runs++;
    const R5900 &rf = *vu.regs;
    int vfbad = -1, vibad = -1; bool accbad = memcmp(&rf.vacc, &r1.vacc, 16) != 0, qbad = memcmp(&rf.vq, &r1.vq, 4) != 0;
    for (int i = 0; i < 32 && vfbad < 0; i++) if (memcmp(&rf.vf[i], &r1.vf[i], 16)) vfbad = i;
    for (int i = 0; i < 32 && vibad < 0; i++) if (rf.vi[i] != r1.vi[i]) vibad = i;
    int membad = -1;
    for (size_t i = 0; i < m1.size(); i++) if (vu.mem[i] != m1[i]) { membad = (int)i; break; }
    if (vfbad >= 0 || vibad >= 0 || accbad || qbad || membad >= 0 || vu.cycle != s1.cycle) {
        if (++bad <= 40)
            rt_log(LOG_WARN, "VUCHECK VU%d start %04X: vf%d vi%d acc%d q%d mem%X cycle %lld/%lld (run %llu)", vu.index, start_pc, vfbad, vibad, accbad, qbad,
                   membad, (long long)vu.cycle, (long long)s1.cycle, (unsigned long long)runs);
        if (vfbad >= 0 && bad <= 40)
            rt_log(LOG_WARN, "   vf%d fast %08X %08X %08X %08X ref %08X %08X %08X %08X", vfbad, rf.vf[vfbad].uw[0], rf.vf[vfbad].uw[1], rf.vf[vfbad].uw[2], rf.vf[vfbad].uw[3],
                   r1.vf[vfbad].uw[0], r1.vf[vfbad].uw[1], r1.vf[vfbad].uw[2], r1.vf[vfbad].uw[3]);
        if (vibad >= 0 && bad <= 40) rt_log(LOG_WARN, "   vi%d fast %08X ref %08X", vibad, rf.vi[vibad], r1.vi[vibad]);
    }
    if (runs % 100000 == 0) rt_log(LOG_INFO, "VUCHECK: %llu VU0 runs, %llu mismatches", (unsigned long long)runs, (unsigned long long)bad);
}
void vu_execute(VuState &vu, uint32_t start_pc) {
    if (vu.index == 1) g_vu1_last_start = start_pc;
    if (vu.index == 1) {   // debug: drop dumpvu.flag in the working dir to dump VU1 micro memory
        static uint32_t n; static auto last = std::chrono::steady_clock::now();
        if (++n % 1024 == 0 && std::chrono::steady_clock::now() - last > std::chrono::seconds(1))
        if (last = std::chrono::steady_clock::now(); FILE *f = fopen("dumpvu.flag", "rb")) {
            fclose(f); remove("dumpvu.flag");
            if (FILE *o = fopen("vu1micro.bin", "wb")) { fwrite(vu.micro, 1, vu.micro_mask + 1, o); fclose(o); }
            rt_log(LOG_INFO, "VU1: dumped micro memory");
        }
    }
    if (vu.index == 1) {   // debug: NPPL_VUDUMP=<start hex> dumps VU1 micro+data on the Nth run of that program (NPPL_VUDUMP_N, default 2000)
        static const char *e = getenv("NPPL_VUDUMP");
        static uint32_t want = e ? strtoul(e, nullptr, 16) : ~0u, left = getenv("NPPL_VUDUMP_N") ? atoi(getenv("NPPL_VUDUMP_N")) : 2000;
        if (start_pc == want && left && !--left) {
            if (FILE *o = fopen("vu1dump_micro.bin", "wb")) { fwrite(vu.micro, 1, vu.micro_mask + 1, o); fclose(o); }
            if (FILE *o = fopen("vu1dump_mem.bin", "wb")) { fwrite(vu.mem, 1, vu.mem_mask + 1, o); fclose(o); }
            if (FILE *o = fopen("vu1dump_regs.bin", "wb")) { fwrite(vu.regs->vf, 16, 32, o); fwrite(vu.regs->vi, 4, 32, o); fclose(o); }
            rt_log(LOG_INFO, "VU1: dumped state at start %04X", start_pc);
        }
    }
    if (vu.index == 1) {   // debug: NPPL_VUSNAP=<n> saves the inputs of VU1 runs n..n+255 to vusnap.bin (tools/vubench replays them)
        static const char *e = getenv("NPPL_VUSNAP");
        static uint64_t at = e ? strtoull(e, nullptr, 0) : ~0ull, count;
        static FILE *snap;
        if (e && count >= at && count < at + 256) {
            if (!snap) snap = fopen("vusnap.bin", "wb");
            if (snap) {
                fwrite(&start_pc, 4, 1, snap); fwrite(&vu, sizeof vu, 1, snap); fwrite(vu.regs, sizeof *vu.regs, 1, snap);
                fwrite(vu.micro, 1, 0x4000, snap); fwrite(vu.mem, 1, 0x4000, snap);
                if (count + 1 == at + 256) { fclose(snap); snap = nullptr; rt_log(LOG_INFO, "VU1: wrote vusnap.bin"); }
            }
        }
        count++;
    }
    auto t0 = std::chrono::steady_clock::now();
    static const bool jit = getenv("NPPL_NOJIT") == nullptr;
    static const bool check = getenv("NPPL_VUCHECK") != nullptr;
    if (check && vu.index == 0) vu_check_run(vu, start_pc);
    else if (jit && vu.index == 1) vu_execute_jit(vu, start_pc); else vu_execute_impl(vu, start_pc);
    g_prof_vu_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}
static void vu_execute_impl(VuState &vu, uint32_t start_pc) {
    R5900 *r = vu.regs;
    uint32_t pc = start_pc & vu.micro_mask;
    bool branch_pending = false, ending = false;
    uint32_t branch_target = 0;
    for (uint64_t n = 0; n < 4000000; n++) {
        g_vu_instr++;
        vu.cycle++;
        if (vu.q_ready && vu.cycle >= vu.q_ready) { r->vq = vu.q_pending; vu.q_ready = 0; }
        if (vu.p_ready && vu.cycle >= vu.p_ready) { vu.p = vu.p_pending; vu.p_ready = 0; }
        uint32_t lo, up;
        memcpy(&lo, vu.micro + pc, 4);
        memcpy(&up, vu.micro + pc + 4, 4);
        if (up & 0x80000000u) r->vi[VI_I] = lo;        // I bit: lower word is a float immediate
        // Pipeline timing: stall on FMAC data hazards (results usable 4 cycles after issue) and on
        // WAITQ/WAITP or a new FDIV/EFU op while one is in flight. Microcode schedules DIV/Q use
        // around these stalls, so Q/P availability must follow the stalled cycle count.
        VuDec &dc = g_dec[vu.index][(pc >> 3) & 2047];
        if (!dc.valid || dc.lo != lo || dc.up != up) {
            decode_pair(dc, lo, up);

        }
        {
            int64_t *ready = &vu.vf_ready[0][0];
            int64_t need = vu.cycle;
            for (int i = 0; i < dc.nrd; i++) need = std::max(need, ready[dc.rd[i]]);
            if (dc.wait_q && vu.q_ready) need = std::max(need, vu.q_ready);
            if (dc.wait_p && vu.p_ready) need = std::max(need, vu.p_ready);
            if (need > vu.cycle) { vu.cycle = need; commit_q(vu, false); commit_p(vu, false); }
            for (int i = 0; i < dc.nwr; i++) ready[dc.wr[i]] = vu.cycle + 4;
        }
        uint32_t br = 0;
        int taken;
        if (dc.ud) {
            taken = dc.lower_nop ? 0 : dc.l.fn(vu, dc.l, lo, pc, &br);
            dc.ud(vu, dc.u);
        } else {
            UpperResult u;
            dc.u.fn(vu, dc.u, up, u);
            taken = dc.lower_nop ? 0 : dc.l.fn(vu, dc.l, lo, pc, &br);
            commit_upper_fast(vu, u);
        }
        r->vf[0] = VF0; r->vi[0] = 0;
        bool was_branch = branch_pending, was_ending = ending;
        uint32_t target = branch_target;
        if (up & 0x40000000u) ending = true;              // E bit: stop after the next pair
        if (taken) { branch_pending = true; branch_target = br; }
        else if (was_branch) branch_pending = false;
        pc = (pc + 8) & vu.micro_mask;
        if (was_branch) { pc = target; if (!taken) branch_pending = false; }
        if (was_ending) { vu.tpc = pc; r->vi[VI_TPC] = pc / 8; return; }
    }
    rt_log(LOG_WARN, "VU%d: program at %04X did not finish (runaway?)", vu.index, start_pc);
}
