// Graphics Synthesizer: GIF paths, GS registers, software rasterizer, display readout.
#pragma once
#include <cstdint>

enum { GIF_PATH1 = 0, GIF_PATH2 = 1, GIF_PATH3 = 2 };

void gs_init();
// Feed qwords to a GIF path (path 1 = VU1 XGKICK, 2 = VIF1 DIRECT, 3 = GIF DMA).
void gif_feed(int path, const uint8_t *data, uint32_t qwords);
// Convert the displayed frame buffer (per PMODE/DISPFB/DISPLAY) to 32-bit BGRA.
bool gs_read_display(uint32_t *out, int max_w, int max_h, int &w, int &h);
uint64_t gs_priv_reg(uint32_t off);
void gs_set_csr_bits(uint32_t bits);
