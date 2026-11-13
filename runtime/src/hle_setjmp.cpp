// setjmp/longjmp for recompiled code.
// Each guest jmp_buf address gets a host jmp_buf. The recompiler wraps every
// `jal setjmp` in a host setjmp on that buffer; hle_longjmp restores the
// callee-saved guest registers from the guest buffer (exactly like the original
// libc longjmp at 0x107670) and then host-longjmps back to the setjmp call site.
#include "rt_internal.h"
#include <csetjmp>
#include <cstdlib>
#include <unordered_map>

static std::unordered_map<uint32_t, void *> g_jbs;

extern "C" void *rt_jmpbuf(uint32_t guest_buf) {
    void *&p = g_jbs[guest_buf];
    if (!p) {
        p = _aligned_malloc(sizeof(jmp_buf), 16);
        memset(p, 0, sizeof(jmp_buf));
    }
    return p;
}

extern "C" void hle_longjmp(R5900 *c) {
    uint32_t a = (uint32_t)c->r[4].ud[0];
    int32_t val = (int32_t)c->r[5].ud[0];
    auto it = g_jbs.find(a);
    if (it == g_jbs.end()) {
        rt_log(LOG_ERR, "longjmp(%08X): no matching setjmp", a);
        rt_exit(1);
    }
    static const int regs[] = {16, 17, 18, 19, 20, 21, 22, 23, 30};
    for (int k = 0; k < 9; k++) LQ(&c->r[regs[k]], a + 16 * k);
    for (int k = 0; k < 8; k++) c->f[24 + k].u = LW(a + 0x90 + 4 * k);
    LQ(&c->r[29], a + 0xB0);
    LQ(&c->r[31], a + 0xC0);
    c->r[2].ud[0] = (uint64_t)(int64_t)(val ? val : 1);
    longjmp(*(jmp_buf *)it->second, 1);
}
