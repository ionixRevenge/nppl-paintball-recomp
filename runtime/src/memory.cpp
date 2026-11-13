// Guest memory map: RAM mirrors, scratchpad, VU memories, hardware registers.
#include "rt_internal.h"
#include <cstdlib>

uint8_t *g_ram;
uint8_t g_spr[0x4000];
uint8_t g_vu0_mem[0x1000], g_vu0_micro[0x1000];
uint8_t g_vu1_mem[0x4000], g_vu1_micro[0x4000];
uint8_t g_iop_ram[0x200000];

// Resolve an address to host memory, or nullptr for MMIO / unmapped.
static uint8_t *resolve(uint32_t a) {
    uint32_t seg = a >> 28;
    if (seg == 0x7 && (a & 0xFFFFC000u) == 0x70000000u) return g_spr + (a & 0x3FFF);
    uint32_t p;
    if (seg >= 0x8 && seg <= 0xB) p = a & 0x1FFFFFFFu;  // kseg0/kseg1
    else if (seg == 0x2 || seg == 0x3) p = a & 0x0FFFFFFFu; // uncached / uncached-accelerated
    else p = a;
    if (p < RAM_SIZE) return g_ram + p;
    if (p >= 0x11000000u && p < 0x11001000u) return g_vu0_micro + (p & 0xFFF);
    if (p >= 0x11004000u && p < 0x11005000u) return g_vu0_mem + (p & 0xFFF);
    if (p >= 0x11008000u && p < 0x1100C000u) return g_vu1_micro + (p & 0x3FFF);
    if (p >= 0x1100C000u && p < 0x11010000u) return g_vu1_mem + (p & 0x3FFF);
    if (p >= 0x1C000000u && p < 0x1C200000u) return g_iop_ram + (p & 0x1FFFFF);
    return nullptr;
}

static bool is_mmio(uint32_t a) {
    uint32_t p = a & 0x1FFFFFFFu;
    return (p >= 0x10000000u && p < 0x10010000u) || (p >= 0x12000000u && p < 0x12002000u);
}

static void unmapped(const char *what, uint32_t a) {
    if (rt_once(0x100000000ull | a))
        rt_log(LOG_WARN, "unmapped %s at %08X", what, a);
}

extern "C" {

uint8_t rt_read8(uint32_t a) {
    if (uint8_t *p = resolve(a)) return *p;
    if (is_mmio(a)) return (uint8_t)(hw_read32(a & ~3u) >> ((a & 3) * 8));
    unmapped("read8", a); return 0;
}
uint16_t rt_read16(uint32_t a) {
    if (uint8_t *p = resolve(a)) { uint16_t v; memcpy(&v, p, 2); return v; }
    if (is_mmio(a)) return (uint16_t)(hw_read32(a & ~3u) >> ((a & 2) * 8));
    unmapped("read16", a); return 0;
}
uint32_t rt_read32(uint32_t a) {
    if (uint8_t *p = resolve(a)) { uint32_t v; memcpy(&v, p, 4); return v; }
    if (is_mmio(a)) return hw_read32(a);
    unmapped("read32", a); return 0;
}
uint64_t rt_read64(uint32_t a) {
    if (uint8_t *p = resolve(a)) { uint64_t v; memcpy(&v, p, 8); return v; }
    if (is_mmio(a)) return hw_read64(a);
    unmapped("read64", a); return 0;
}
void rt_read128(uint32_t a, R128 *v) {
    if (uint8_t *p = resolve(a)) { memcpy(v, p, 16); return; }
    if (is_mmio(a)) { v->ud[0] = hw_read64(a); v->ud[1] = hw_read64(a + 8); return; }
    unmapped("read128", a); memset(v, 0, 16);
}
void rt_write8(uint32_t a, uint8_t v) {
    if (uint8_t *p = resolve(a)) { *p = v; return; }
    if (is_mmio(a)) { hw_write32(a & ~3u, (uint32_t)v << ((a & 3) * 8)); return; }
    unmapped("write8", a);
}
void rt_write16(uint32_t a, uint16_t v) {
    if (uint8_t *p = resolve(a)) { memcpy(p, &v, 2); return; }
    if (is_mmio(a)) { hw_write32(a & ~3u, (uint32_t)v << ((a & 2) * 8)); return; }
    unmapped("write16", a);
}
void rt_write32(uint32_t a, uint32_t v) {
    if (uint8_t *p = resolve(a)) { memcpy(p, &v, 4); return; }
    if (is_mmio(a)) { hw_write32(a, v); return; }
    unmapped("write32", a);
}
void rt_write64(uint32_t a, uint64_t v) {
    if (uint8_t *p = resolve(a)) { memcpy(p, &v, 8); return; }
    if (is_mmio(a)) { hw_write64(a, v); return; }
    unmapped("write64", a);
}
void rt_write128(uint32_t a, const R128 *v) {
    if (uint8_t *p = resolve(a)) { memcpy(p, v, 16); return; }
    if (is_mmio(a)) { hw_write64(a, v->ud[0]); hw_write64(a + 8, v->ud[1]); return; }
    unmapped("write128", a);
}

} // extern "C"
