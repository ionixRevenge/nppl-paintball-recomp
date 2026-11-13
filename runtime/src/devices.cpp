// Small device glue: vblank -> display, GS CSR, VU0 micro calls from the EE.
#include "rt_internal.h"
#include "hw.h"
#include "gs.h"
#include "vu.h"

void display_vsync();          // display.cpp
void gs_csr_ack(uint32_t v);   // gs.cpp

uint32_t g_vsync_count;
void pad_vsync();             // pad.cpp
void hw_vsync_tick() { g_vsync_count++; pad_vsync(); display_vsync(); }

void gs_csr_write(uint32_t v) {
    if (v & 0x200) rt_log(LOG_INFO, "GS reset");
    gs_csr_ack(v);
}

void vu_fbrst(uint32_t) {}

// VCALLMS / VCALLMSR: VU0 micro mode runs on the EE's COP2 register file.
void vu0_run_micro(R5900 *c, uint32_t addr) {
    g_vu0.regs = c;
    vu_execute(g_vu0, addr);
}
