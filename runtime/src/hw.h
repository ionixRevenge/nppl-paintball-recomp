// Hardware register addresses and device hooks.
#pragma once
#include <cstdint>

enum : uint32_t {
    T0_COUNT = 0x10000000, T1_COUNT = 0x10000800, T2_COUNT = 0x10001000, T3_COUNT = 0x10001800,
    GIF_STAT = 0x10003020,
    VIF0_STAT = 0x10003800, VIF1_STAT = 0x10003C00,
    GIF_FIFO = 0x10006000,
    D_CTRL = 0x1000E000, D_STAT = 0x1000E010,
    INTC_STAT = 0x1000F000, INTC_MASK = 0x1000F010,
    SIF_MSCOM = 0x1000F200, SIF_SMCOM = 0x1000F210, SIF_MSFLG = 0x1000F220, SIF_SMFLG = 0x1000F230,
    D_ENABLER = 0x1000F520, D_ENABLEW = 0x1000F590,
    GS_CSR = 0x12001000,
};

enum { INTC_GS = 0, INTC_SBUS = 1, INTC_VBLANK_START = 2, INTC_VBLANK_END = 3,
       INTC_VIF0 = 4, INTC_VIF1 = 5, INTC_VU0 = 6, INTC_VU1 = 7, INTC_IPU = 8,
       INTC_TIM0 = 9, INTC_TIM1 = 10, INTC_TIM2 = 11, INTC_TIM3 = 12 };

uint32_t hw_reg(uint32_t p);
void hw_set_reg(uint32_t p, uint32_t v);
void hw_raise_intc(int line);
uint32_t hw_timer_poll();   // INTC timer lines that fired since the last poll

// device hooks (dma.cpp / gs.cpp / sif.cpp)
void dma_transfer(int ch, uint32_t chcr, uint32_t madr, uint32_t qwc, uint32_t tadr);
void path3_idle();  // vblank: release PATH3 data left unmasked
void gif_fifo_write32(uint32_t v);
void gs_csr_write(uint32_t v);
void gs_priv_write(uint32_t off, uint64_t v);
void sif_flag_write(uint32_t reg, uint32_t v);

// kernel hooks
void kernel_safepoint();
void kernel_dmac_event(int ch);
