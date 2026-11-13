// Graphics Synthesizer emulation: GIF paths, GS registers, local memory (with the
// real block/column swizzle), image transfers, CLUT and a software rasterizer.
#include "rt_internal.h"
#include "gs.h"
#include "hw.h"
#define NOMINMAX
#include <windows.h>
#pragma comment(lib, "Synchronization.lib")
#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>
#include <string>
#include <condition_variable>
#include <immintrin.h>
#include <thread>
#include <unordered_map>
#include <chrono>
#include <array>
#include <utility>
extern uint64_t g_prof_draw_ns;

// ================================================================ local memory
static uint32_t *g_vram;   // 4 MB
static const uint32_t VRAM_WORDS = 1u << 20;

enum PSM { CT32 = 0x00, CT24 = 0x01, CT16 = 0x02, CT16S = 0x0A, T8 = 0x13, T4 = 0x14, T8H = 0x1B,
           T4HL = 0x24, T4HH = 0x2C, Z32 = 0x30, Z24 = 0x31, Z16 = 0x32, Z16S = 0x3A };

static const uint8_t blockTable32[4][8] = {
    {0, 1, 4, 5, 16, 17, 20, 21}, {2, 3, 6, 7, 18, 19, 22, 23},
    {8, 9, 12, 13, 24, 25, 28, 29}, {10, 11, 14, 15, 26, 27, 30, 31}};
static const uint8_t blockTableZ32[4][8] = {
    {24, 25, 28, 29, 8, 9, 12, 13}, {26, 27, 30, 31, 10, 11, 14, 15},
    {16, 17, 20, 21, 0, 1, 4, 5}, {18, 19, 22, 23, 2, 3, 6, 7}};
static const uint8_t blockTable16[8][4] = {
    {0, 2, 8, 10}, {1, 3, 9, 11}, {4, 6, 12, 14}, {5, 7, 13, 15},
    {16, 18, 24, 26}, {17, 19, 25, 27}, {20, 22, 28, 30}, {21, 23, 29, 31}};
static const uint8_t blockTable16S[8][4] = {
    {0, 2, 16, 18}, {1, 3, 17, 19}, {8, 10, 24, 26}, {9, 11, 25, 27},
    {4, 6, 20, 22}, {5, 7, 21, 23}, {12, 14, 28, 30}, {13, 15, 29, 31}};
static const uint8_t blockTableZ16[8][4] = {
    {24, 26, 8, 10}, {25, 27, 9, 11}, {28, 30, 12, 14}, {29, 31, 13, 15},
    {16, 18, 0, 2}, {17, 19, 1, 3}, {20, 22, 4, 6}, {21, 23, 5, 7}};
static const uint8_t blockTableZ16S[8][4] = {
    {24, 26, 16, 18}, {25, 27, 17, 19}, {8, 10, 0, 2}, {9, 11, 1, 3},
    {28, 30, 20, 22}, {29, 31, 21, 23}, {12, 14, 4, 6}, {13, 15, 5, 7}};
static const uint8_t blockTable8[4][8] = {
    {0, 1, 4, 5, 16, 17, 20, 21}, {2, 3, 6, 7, 18, 19, 22, 23},
    {8, 9, 12, 13, 24, 25, 28, 29}, {10, 11, 14, 15, 26, 27, 30, 31}};
static const uint8_t blockTable4[8][4] = {
    {0, 2, 8, 10}, {1, 3, 9, 11}, {4, 6, 12, 14}, {5, 7, 13, 15},
    {16, 18, 24, 26}, {17, 19, 25, 27}, {20, 22, 28, 30}, {21, 23, 29, 31}};

static const uint8_t columnTable32[8][8] = {
    {0, 1, 4, 5, 8, 9, 12, 13}, {2, 3, 6, 7, 10, 11, 14, 15},
    {16, 17, 20, 21, 24, 25, 28, 29}, {18, 19, 22, 23, 26, 27, 30, 31},
    {32, 33, 36, 37, 40, 41, 44, 45}, {34, 35, 38, 39, 42, 43, 46, 47},
    {48, 49, 52, 53, 56, 57, 60, 61}, {50, 51, 54, 55, 58, 59, 62, 63}};
static const uint8_t columnTable16[8][16] = {
    {0, 2, 8, 10, 16, 18, 24, 26, 1, 3, 9, 11, 17, 19, 25, 27},
    {4, 6, 12, 14, 20, 22, 28, 30, 5, 7, 13, 15, 21, 23, 29, 31},
    {32, 34, 40, 42, 48, 50, 56, 58, 33, 35, 41, 43, 49, 51, 57, 59},
    {36, 38, 44, 46, 52, 54, 60, 62, 37, 39, 45, 47, 53, 55, 61, 63},
    {64, 66, 72, 74, 80, 82, 88, 90, 65, 67, 73, 75, 81, 83, 89, 91},
    {68, 70, 76, 78, 84, 86, 92, 94, 69, 71, 77, 79, 85, 87, 93, 95},
    {96, 98, 104, 106, 112, 114, 120, 122, 97, 99, 105, 107, 113, 115, 121, 123},
    {100, 102, 108, 110, 116, 118, 124, 126, 101, 103, 109, 111, 117, 119, 125, 127}};
static const uint8_t columnTable8[16][16] = {
    {0, 4, 16, 20, 32, 36, 48, 52, 2, 6, 18, 22, 34, 38, 50, 54},
    {8, 12, 24, 28, 40, 44, 56, 60, 10, 14, 26, 30, 42, 46, 58, 62},
    {33, 37, 49, 53, 1, 5, 17, 21, 35, 39, 51, 55, 3, 7, 19, 23},
    {41, 45, 57, 61, 9, 13, 25, 29, 43, 47, 59, 63, 11, 15, 27, 31},
    {96, 100, 112, 116, 64, 68, 80, 84, 98, 102, 114, 118, 66, 70, 82, 86},
    {104, 108, 120, 124, 72, 76, 88, 92, 106, 110, 122, 126, 74, 78, 90, 94},
    {65, 69, 81, 85, 97, 101, 113, 117, 67, 71, 83, 87, 99, 103, 115, 119},
    {73, 77, 89, 93, 105, 109, 121, 125, 75, 79, 91, 95, 107, 111, 123, 127},
    {128, 132, 144, 148, 160, 164, 176, 180, 130, 134, 146, 150, 162, 166, 178, 182},
    {136, 140, 152, 156, 168, 172, 184, 188, 138, 142, 154, 158, 170, 174, 186, 190},
    {161, 165, 177, 181, 129, 133, 145, 149, 163, 167, 179, 183, 131, 135, 147, 151},
    {169, 173, 185, 189, 137, 141, 153, 157, 171, 175, 187, 191, 139, 143, 155, 159},
    {224, 228, 240, 244, 192, 196, 208, 212, 226, 230, 242, 246, 194, 198, 210, 214},
    {232, 236, 248, 252, 200, 204, 216, 220, 234, 238, 250, 254, 202, 206, 218, 222},
    {193, 197, 209, 213, 225, 229, 241, 245, 195, 199, 211, 215, 227, 231, 243, 247},
    {201, 205, 217, 221, 233, 237, 249, 253, 203, 207, 219, 223, 235, 239, 251, 255}};
static const uint16_t columnTable4[16][32] = {
    {0, 8, 32, 40, 64, 72, 96, 104, 2, 10, 34, 42, 66, 74, 98, 106, 4, 12, 36, 44, 68, 76, 100, 108, 6, 14, 38, 46, 70, 78, 102, 110},
    {16, 24, 48, 56, 80, 88, 112, 120, 18, 26, 50, 58, 82, 90, 114, 122, 20, 28, 52, 60, 84, 92, 116, 124, 22, 30, 54, 62, 86, 94, 118, 126},
    {65, 73, 97, 105, 1, 9, 33, 41, 67, 75, 99, 107, 3, 11, 35, 43, 69, 77, 101, 109, 5, 13, 37, 45, 71, 79, 103, 111, 7, 15, 39, 47},
    {81, 89, 113, 121, 17, 25, 49, 57, 83, 91, 115, 123, 19, 27, 51, 59, 85, 93, 117, 125, 21, 29, 53, 61, 87, 95, 119, 127, 23, 31, 55, 63},
    {192, 200, 224, 232, 128, 136, 160, 168, 194, 202, 226, 234, 130, 138, 162, 170, 196, 204, 228, 236, 132, 140, 164, 172, 198, 206, 230, 238, 134, 142, 166, 174},
    {208, 216, 240, 248, 144, 152, 176, 184, 210, 218, 242, 250, 146, 154, 178, 186, 212, 220, 244, 252, 148, 156, 180, 188, 214, 222, 246, 254, 150, 158, 182, 190},
    {129, 137, 161, 169, 193, 201, 225, 233, 131, 139, 163, 171, 195, 203, 227, 235, 133, 141, 165, 173, 197, 205, 229, 237, 135, 143, 167, 175, 199, 207, 231, 239},
    {145, 153, 177, 185, 209, 217, 241, 249, 147, 155, 179, 187, 211, 219, 243, 251, 149, 157, 181, 189, 213, 221, 245, 253, 151, 159, 183, 191, 215, 223, 247, 255},
    {256, 264, 288, 296, 320, 328, 352, 360, 258, 266, 290, 298, 322, 330, 354, 362, 260, 268, 292, 300, 324, 332, 356, 364, 262, 270, 294, 302, 326, 334, 358, 366},
    {272, 280, 304, 312, 336, 344, 368, 376, 274, 282, 306, 314, 338, 346, 370, 378, 276, 284, 308, 316, 340, 348, 372, 380, 278, 286, 310, 318, 342, 350, 374, 382},
    {321, 329, 353, 361, 257, 265, 289, 297, 323, 331, 355, 363, 259, 267, 291, 299, 325, 333, 357, 365, 261, 269, 293, 301, 327, 335, 359, 367, 263, 271, 295, 303},
    {337, 345, 369, 377, 273, 281, 305, 313, 339, 347, 371, 379, 275, 283, 307, 315, 341, 349, 373, 381, 277, 285, 309, 317, 343, 351, 375, 383, 279, 287, 311, 319},
    {448, 456, 480, 488, 384, 392, 416, 424, 450, 458, 482, 490, 386, 394, 418, 426, 452, 460, 484, 492, 388, 396, 420, 428, 454, 462, 486, 494, 390, 398, 422, 430},
    {464, 472, 496, 504, 400, 408, 432, 440, 466, 474, 498, 506, 402, 410, 434, 442, 468, 476, 500, 508, 404, 412, 436, 444, 470, 478, 502, 510, 406, 414, 438, 446},
    {385, 393, 417, 425, 449, 457, 481, 489, 387, 395, 419, 427, 451, 459, 483, 491, 389, 397, 421, 429, 453, 461, 485, 493, 391, 399, 423, 431, 455, 463, 487, 495},
    {401, 409, 433, 441, 465, 473, 497, 505, 403, 411, 435, 443, 467, 475, 499, 507, 405, 413, 437, 445, 469, 477, 501, 509, 407, 415, 439, 447, 471, 479, 503, 511}};

// Offset of each pixel inside its page (block table + column table folded together), built by
// build_swizzle(): [0] CT32, [1] Z32 (words); 16-bit: [0] CT16, [1] CT16S, [2] Z16, [3] Z16S (halfwords).
static uint16_t g_swz32[2][32][64], g_swz16[4][64][64];
static void build_swizzle() {
    const uint8_t (*b32[2])[8] = {blockTable32, blockTableZ32};
    const uint8_t (*b16[4])[4] = {blockTable16, blockTable16S, blockTableZ16, blockTableZ16S};
    for (int t = 0; t < 2; t++)
        for (int y = 0; y < 32; y++)
            for (int x = 0; x < 64; x++) g_swz32[t][y][x] = (uint16_t)(b32[t][(y >> 3) & 3][(x >> 3) & 7] * 64 + columnTable32[y & 7][x & 7]);
    for (int t = 0; t < 4; t++)
        for (int y = 0; y < 64; y++)
            for (int x = 0; x < 64; x++) g_swz16[t][y][x] = (uint16_t)(b16[t][(y >> 3) & 7][(x >> 4) & 3] * 128 + columnTable16[y & 7][x & 15]);
}
// bp: base in blocks (64 words); bw: buffer width in units of 64 pixels
static inline uint32_t addr32(uint32_t bp, uint32_t bw, uint32_t x, uint32_t y, const uint8_t (*bt)[8]) {
    const uint16_t (*sw)[64] = g_swz32[bt == blockTableZ32 ? 1 : 0];
    return (bp * 64 + ((x >> 6) + (y >> 5) * std::max(bw, 1u)) * 2048 + sw[y & 31][x & 63]) & (VRAM_WORDS - 1);
}
static inline uint32_t addr16(uint32_t bp, uint32_t bw, uint32_t x, uint32_t y, const uint8_t (*bt)[4]) {  // halfword index
    const uint16_t (*sw)[64] = g_swz16[bt == blockTable16S ? 1 : bt == blockTableZ16 ? 2 : bt == blockTableZ16S ? 3 : 0];
    return (bp * 128 + ((x >> 6) + (y >> 6) * std::max(bw, 1u)) * 4096 + sw[y & 63][x & 63]) & (VRAM_WORDS * 2 - 1);
}
static inline uint32_t addr8(uint32_t bp, uint32_t bw, uint32_t x, uint32_t y) {  // byte index
    uint32_t page = (x >> 7) + (y >> 6) * std::max(bw >> 1, 1u);
    uint32_t block = blockTable8[(y >> 4) & 3][(x >> 4) & 7];
    return ((bp + page * 32 + block) * 256 + columnTable8[y & 15][x & 15]) & (VRAM_WORDS * 4 - 1);
}
static inline uint32_t addr4(uint32_t bp, uint32_t bw, uint32_t x, uint32_t y) {  // nibble index
    uint32_t page = (x >> 7) + (y >> 7) * std::max(bw >> 1, 1u);
    uint32_t block = blockTable4[(y >> 4) & 7][(x >> 5) & 3];
    return ((bp + page * 32 + block) * 512 + columnTable4[y & 15][x & 31]) & (VRAM_WORDS * 8 - 1);
}

static inline uint16_t *vram16() { return (uint16_t *)g_vram; }
static inline uint8_t *vram8() { return (uint8_t *)g_vram; }

// raw pixel read/write in a given PSM (value in the format's own bits)
static uint32_t read_raw(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y) {
    switch (psm) {
    case CT32: case CT24: return g_vram[addr32(bp, bw, x, y, blockTable32)];
    case Z32: case Z24: return g_vram[addr32(bp, bw, x, y, blockTableZ32)];
    case CT16: return vram16()[addr16(bp, bw, x, y, blockTable16)];
    case CT16S: return vram16()[addr16(bp, bw, x, y, blockTable16S)];
    case Z16: return vram16()[addr16(bp, bw, x, y, blockTableZ16)];
    case Z16S: return vram16()[addr16(bp, bw, x, y, blockTableZ16S)];
    case T8: return vram8()[addr8(bp, bw, x, y)];
    case T4: { uint32_t n = addr4(bp, bw, x, y); return (vram8()[n >> 1] >> ((n & 1) * 4)) & 15; }
    case T8H: return g_vram[addr32(bp, bw, x, y, blockTable32)] >> 24;
    case T4HL: return (g_vram[addr32(bp, bw, x, y, blockTable32)] >> 24) & 15;
    case T4HH: return g_vram[addr32(bp, bw, x, y, blockTable32)] >> 28;
    }
    return 0;
}
// Write generations per 8 KB page, so decoded textures know when they went stale.
static uint32_t g_page_gen[VRAM_WORDS / 2048];
static uint32_t g_write_gen = 1;
static inline void mark_word(uint32_t wi) { g_page_gen[(wi >> 11) & (VRAM_WORDS / 2048 - 1)] = g_write_gen; }

static void write_raw(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y, uint32_t v) {
    switch (psm) {
    case CT32: case Z32: case CT24: case Z24: case T8H: case T4HL: case T4HH:
        mark_word(addr32(bp, bw, x, y, (psm == Z32 || psm == Z24) ? blockTableZ32 : blockTable32)); break;
    case CT16: mark_word(addr16(bp, bw, x, y, blockTable16) >> 1); break;
    case CT16S: mark_word(addr16(bp, bw, x, y, blockTable16S) >> 1); break;
    case Z16: mark_word(addr16(bp, bw, x, y, blockTableZ16) >> 1); break;
    case Z16S: mark_word(addr16(bp, bw, x, y, blockTableZ16S) >> 1); break;
    case T8: mark_word(addr8(bp, bw, x, y) >> 2); break;
    case T4: mark_word(addr4(bp, bw, x, y) >> 3); break;
    }
    switch (psm) {
    case CT32: case Z32: g_vram[addr32(bp, bw, x, y, psm == Z32 ? blockTableZ32 : blockTable32)] = v; return;
    case CT24: case Z24: { uint32_t &d = g_vram[addr32(bp, bw, x, y, psm == Z24 ? blockTableZ32 : blockTable32)]; d = (d & 0xFF000000u) | (v & 0xFFFFFF); return; }
    case CT16: vram16()[addr16(bp, bw, x, y, blockTable16)] = (uint16_t)v; return;
    case CT16S: vram16()[addr16(bp, bw, x, y, blockTable16S)] = (uint16_t)v; return;
    case Z16: vram16()[addr16(bp, bw, x, y, blockTableZ16)] = (uint16_t)v; return;
    case Z16S: vram16()[addr16(bp, bw, x, y, blockTableZ16S)] = (uint16_t)v; return;
    case T8: vram8()[addr8(bp, bw, x, y)] = (uint8_t)v; return;
    case T4: { uint32_t n = addr4(bp, bw, x, y); uint8_t &b = vram8()[n >> 1]; b = (n & 1) ? (uint8_t)((b & 0x0F) | (v << 4)) : (uint8_t)((b & 0xF0) | (v & 15)); return; }
    case T8H: { uint32_t &d = g_vram[addr32(bp, bw, x, y, blockTable32)]; d = (d & 0x00FFFFFFu) | (v << 24); return; }
    case T4HL: { uint32_t &d = g_vram[addr32(bp, bw, x, y, blockTable32)]; d = (d & 0xF0FFFFFFu) | ((v & 15) << 24); return; }
    case T4HH: { uint32_t &d = g_vram[addr32(bp, bw, x, y, blockTable32)]; d = (d & 0x0FFFFFFFu) | ((v & 15) << 28); return; }
    }
}
static int psm_bits(uint32_t psm) {
    switch (psm) {
    case CT32: case Z32: return 32;
    case CT24: case Z24: return 24;
    case CT16: case CT16S: case Z16: case Z16S: return 16;
    case T8: case T8H: return 8;
    case T4: case T4HL: case T4HH: return 4;
    }
    return 32;
}

// ================================================================ registers
static uint64_t g_reg[0x100];
static uint64_t g_priv[0x2000 / 8];
static uint32_t g_csr_events;

enum {
    R_PRIM = 0x00, R_RGBAQ = 0x01, R_ST = 0x02, R_UV = 0x03, R_XYZF2 = 0x04, R_XYZ2 = 0x05, R_TEX0_1 = 0x06,
    R_TEX0_2 = 0x07, R_CLAMP_1 = 0x08, R_CLAMP_2 = 0x09, R_FOG = 0x0A, R_XYZF3 = 0x0C, R_XYZ3 = 0x0D,
    R_TEX1_1 = 0x14, R_TEX1_2 = 0x15, R_MIPTBP1_1 = 0x34, R_MIPTBP1_2 = 0x35, R_MIPTBP2_1 = 0x36, R_MIPTBP2_2 = 0x37, R_TEX2_1 = 0x16, R_TEX2_2 = 0x17, R_XYOFFSET_1 = 0x18, R_XYOFFSET_2 = 0x19,
    R_PRMODECONT = 0x1A, R_PRMODE = 0x1B, R_TEXCLUT = 0x1C, R_SCANMSK = 0x22, R_TEXA = 0x3B, R_FOGCOL = 0x3D,
    R_TEXFLUSH = 0x3F, R_SCISSOR_1 = 0x40, R_SCISSOR_2 = 0x41, R_ALPHA_1 = 0x42, R_ALPHA_2 = 0x43, R_DIMX = 0x44,
    R_DTHE = 0x45, R_COLCLAMP = 0x46, R_TEST_1 = 0x47, R_TEST_2 = 0x48, R_PABE = 0x49, R_FBA_1 = 0x4A, R_FBA_2 = 0x4B,
    R_FRAME_1 = 0x4C, R_FRAME_2 = 0x4D, R_ZBUF_1 = 0x4E, R_ZBUF_2 = 0x4F, R_BITBLTBUF = 0x50, R_TRXPOS = 0x51,
    R_TRXREG = 0x52, R_TRXDIR = 0x53, R_HWREG = 0x54, R_SIGNAL = 0x60, R_FINISH = 0x61, R_LABEL = 0x62,
};

static inline uint64_t bits(uint64_t v, int lo, int n) { return (v >> lo) & ((1ull << n) - 1); }

static void gs_flush();
static void pxstat_report();
static int g_wait_reason;               // debug: why the game thread waited on the rasterizer
static uint64_t g_wait_ns[8], g_wait_n[8];
// Draw batches rasterize on worker threads while the EE/VU keep going. Pages the queued and the
// in-flight batch may touch (frame and Z buffers) are tracked so anything else that reads or
// writes VRAM only has to wait when it actually overlaps them.
static const uint32_t NPAGES = (1u << 20) / 2048;
static uint8_t g_pg_cur[NPAGES], g_pg_cnt[NPAGES];   // queued batch / number of in-flight batches touching the page
// Render targets sampled straight from VRAM by queued draws (no decode), and pages the GS has drawn to.
static uint8_t g_rd_cur[NPAGES], g_rd_cnt[NPAGES], g_page_drawn[NPAGES];
static bool g_rd_any_cur;
// Ordering inside the queued batch: pages written / read in place since the last in-stream
// barrier. A read-after-write or write-after-read between queued draws only needs the
// rasterizer threads to sync with each other (a barrier command), not the game thread.
static uint8_t g_wr_since[NPAGES], g_rd_since[NPAGES];
static bool g_wr_since_any, g_rd_since_any;
static void push_barrier();
static std::vector<std::vector<uint32_t>> g_grave_cur;   // decoded textures still referenced by queued draws
static inline bool page_busy(uint32_t p) { p &= NPAGES - 1; return g_pg_cur[p] | g_pg_cnt[p] | g_rd_cur[p] | g_rd_cnt[p]; }
static uint32_t page_at(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y);
template <class F> static void for_rect_pages(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h, F f) {
    if (!w || !h) return;
    for (uint32_t y = y0;; y += 8) {
        uint32_t yy = std::min(y, y0 + h - 1);
        for (uint32_t x = x0;; x += 8) {
            uint32_t xx = std::min(x, x0 + w - 1);
            if (f(page_at(psm, bp, bw, xx & 2047, yy & 2047) & (NPAGES - 1))) return;
            if (xx == x0 + w - 1) break;
        }
        if (yy == y0 + h - 1) break;
    }
}
// Wait for the rasterizer if any page of the rectangle is in use by queued draws.
static void reap();
static bool g_gpu;
static void gpu_sync_rect(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h, bool write);
static void sync_rect(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h, int tag = 0) {
    if (!w || !h) return;
    if (g_gpu) gpu_sync_rect(psm, bp, bw, x0, y0, w, h, false);
    reap();
    for (uint32_t y = y0;; y += 8) {
        uint32_t yy = std::min(y, y0 + h - 1);
        for (uint32_t x = x0;; x += 8) {
            uint32_t xx = std::min(x, x0 + w - 1);
            if (page_busy(page_at(psm, bp, bw, xx & 2047, yy & 2047))) {
                static const bool lg = getenv("NPPL_RECTLOG") != nullptr; if (lg)
                    rt_log(LOG_INFO, "rect wait: tag %d psm %02X bp %04X bw %u %u,%u %ux%u page %u cur %d cnt %d rd %d rdcnt %d drawn %d", tag, psm, bp, bw, x0, y0, w, h, page_at(psm, bp, bw, xx & 2047, yy & 2047) & (NPAGES - 1), g_pg_cur[page_at(psm, bp, bw, xx & 2047, yy & 2047) & (NPAGES - 1)], g_pg_cnt[page_at(psm, bp, bw, xx & 2047, yy & 2047) & (NPAGES - 1)], g_rd_cur[page_at(psm, bp, bw, xx & 2047, yy & 2047) & (NPAGES - 1)], g_rd_cnt[page_at(psm, bp, bw, xx & 2047, yy & 2047) & (NPAGES - 1)], g_page_drawn[page_at(psm, bp, bw, xx & 2047, yy & 2047) & (NPAGES - 1)]);
                g_wait_reason = 1; gs_flush(); return;
            }
            if (xx == x0 + w - 1) break;
        }
        if (yy == y0 + h - 1) break;
    }
}

// ================================================================ CLUT
// GS CLUT buffer: 512 16-bit slots. A CT32 entry e keeps its low half in slot e and its high half in
// slot e+256; CT16 entries use one slot each. CSA offsets both loads and reads (see clut_entry).
static uint16_t g_clut[512];
static uint32_t g_clut_gen = 1;  // bumped on every CLUT load
static uint32_t g_cbp0 = ~0u, g_cbp1 = ~0u;

static void clut_load(uint64_t tex0) {
    uint32_t psm = (uint32_t)bits(tex0, 20, 6), cbp = (uint32_t)bits(tex0, 37, 14), cpsm = (uint32_t)bits(tex0, 51, 4);
    uint32_t csm = (uint32_t)bits(tex0, 55, 1), csa = (uint32_t)bits(tex0, 56, 5), cld = (uint32_t)bits(tex0, 61, 3);
    bool load = false;
    switch (cld) {
    case 1: load = true; break;
    case 2: load = true; g_cbp0 = cbp; break;
    case 3: load = true; g_cbp1 = cbp; break;
    case 4: if (cbp != g_cbp0) { load = true; g_cbp0 = cbp; } break;
    case 5: if (cbp != g_cbp1) { load = true; g_cbp1 = cbp; } break;
    }
    if (!load) return;
    bool is8 = psm == T8 || psm == T8H, is4 = psm == T4 || psm == T4HL || psm == T4HH;
    if (!is8 && !is4) return;
    bool c32 = cpsm == 0 || cpsm == 1;
    uint32_t cpsm_mem = c32 ? CT32 : (cpsm == 0x0A ? CT16S : CT16);
    uint64_t texclut = g_reg[R_TEXCLUT];
    if (csm == 0) sync_rect(cpsm_mem, cbp, 1, 0, 0, 16, is8 ? 16 : 2);
    else sync_rect(cpsm_mem, cbp, (uint32_t)bits(texclut, 0, 6), (uint32_t)bits(texclut, 6, 6) * 16, (uint32_t)bits(texclut, 12, 10), is8 ? 256 : 16, 1);
    // CT32 8-bit loads start at block CSA (entries below it keep their old data); 4-bit loads land at CSA*16.
    int first = is8 && c32 ? (int)(csa & 15) * 16 : 0, n = is8 ? 256 : 16;
    for (int i = first; i < n; i++) {
        uint32_t x, y, bw = 1;
        if (csm == 0) {
            if (is8) { uint32_t p = (i & 0xE7) | ((i & 8) << 1) | ((i & 0x10) >> 1); x = p & 15; y = p >> 4; }
            else { x = i & 7; y = i >> 3; }
        } else {
            bw = (uint32_t)bits(texclut, 0, 6);
            x = (uint32_t)bits(texclut, 6, 6) * 16 + i; y = (uint32_t)bits(texclut, 12, 10);
        }
        uint32_t v = read_raw(cpsm_mem, cbp, bw, x, y);
        if (c32) {
            uint32_t e = is8 ? (uint32_t)i : ((csa & 15) * 16 + i);
            g_clut[e] = (uint16_t)v; g_clut[e + 256] = (uint16_t)(v >> 16);
        } else {
            g_clut[(csa * 16 + i) & 511] = (uint16_t)v;
        }
    }
    g_clut_gen++;
}

// ================================================================ texturing
struct Rgba { int r, g, b, a; };

static inline Rgba from16(uint32_t v, uint64_t texa, bool aem) {
    Rgba c{(int)((v & 31) << 3), (int)(((v >> 5) & 31) << 3), (int)(((v >> 10) & 31) << 3), 0};
    if (aem && (v & 0x7FFF) == 0) c.a = 0;
    else c.a = (v & 0x8000) ? (int)bits(texa, 32, 8) : (int)bits(texa, 0, 8);
    return c;
}
static inline Rgba from32(uint32_t v) { return Rgba{(int)(v & 255), (int)((v >> 8) & 255), (int)((v >> 16) & 255), (int)(v >> 24)}; }

struct TexState {
    uint32_t psm, tbp, tbw, tw, th, cpsm, csa, tcc, tfx;
    uint32_t wms, wmt, minu, maxu, minv, maxv;
    uint64_t texa;
    bool bilinear;         // TEX1.MMAG (and the filter used when not mipmapping)
    const uint32_t *pix;   // decoded texture (tw x th, RGBA8888), set by tex_bind; null = sample VRAM directly
    bool direct;
    // Mipmapping (TEX1): levels 1..mxl come from MIPTBP1/2. LOD = log2(1/|Q|) * 2^L + K
    // (or just K with LCM); LOD <= 0 magnifies with MMAG, otherwise MMIN picks the filter/level.
    int nlev;              // 1 = no mipmapping
    bool lcm, min_linear;
    int lshift; float k;
    uint32_t mtbp[7], mtbw[7];
    const uint32_t *lpix[7];
    uint64_t uid;          // identifies the decoded contents (GPU texture cache key)
};

static Rgba clut_lookup(const TexState &t, uint32_t idx) {
    bool aem = bits(t.texa, 15, 1) != 0;
    bool is4 = t.psm == T4 || t.psm == T4HL || t.psm == T4HH;
    if (t.cpsm == 0 || t.cpsm == 1) {
        // 8-bit reads offset by CSA clamp to the last block instead of wrapping (as PCSX2 found).
        uint32_t e = is4 ? (t.csa & 15) * 16 + idx : std::min((idx & 0xF0) + (t.csa & 15) * 16, 240u) + (idx & 15);
        return from32(g_clut[e] | ((uint32_t)g_clut[e + 256] << 16));
    }
    return from16(g_clut[(t.csa * 16 + idx) & 511], t.texa, aem);
}

static Rgba texel_slow(const TexState &t, int u, int v) {
    bool aem = bits(t.texa, 15, 1) != 0;
    switch (t.psm) {
    case CT32: return from32(read_raw(CT32, t.tbp, t.tbw, u, v));
    case CT24: {
        Rgba c = from32(read_raw(CT24, t.tbp, t.tbw, u, v));
        c.a = (aem && !c.r && !c.g && !c.b) ? 0 : (int)bits(t.texa, 0, 8);
        return c;
    }
    case CT16: case CT16S: return from16(read_raw(t.psm, t.tbp, t.tbw, u, v), t.texa, aem);
    default: return clut_lookup(t, read_raw(t.psm, t.tbp, t.tbw, u, v));
    }
}

struct TexKey {
    uint32_t tbp, tbw, psm, tw, th, cpsm, csa, clut_hash; uint64_t texa;
    bool operator==(const TexKey &o) const { return !memcmp(this, &o, sizeof *this); }
};
struct TexKeyHash { size_t operator()(const TexKey &k) const {
    const uint32_t *w = (const uint32_t *)&k; size_t h = 1469598103934665603ull;
    for (size_t i = 0; i < sizeof k / 4; i++) h = (h ^ w[i]) * 1099511628211ull;
    return h; } };
struct TexEntry { uint32_t gen = 0, used = 0; uint64_t hash = 0; std::vector<uint16_t> pages; std::vector<uint32_t> pix; };
// Several textures are streamed through the same VRAM slot each frame, so one key keeps a few
// decoded variants told apart by the hash of their source pages.
struct TexSlot { TexEntry v[4]; int cur = -1; uint32_t tick = 0; bool direct = false; std::vector<uint16_t> rt_pages; uint32_t undraw_seen = 0; };
// Bumped whenever an upload overwrites drawn pages: direct (render-target) slots then recheck
// whether any of their pages still holds GS-drawn data, and go back to decoding if not.
static uint32_t g_undraw_gen = 1;
static bool slot_still_direct(TexSlot &s) {
    if (s.undraw_seen == g_undraw_gen) return true;
    s.undraw_seen = g_undraw_gen;
    for (uint16_t p : s.rt_pages) if (g_page_drawn[p]) return true;
    s.direct = false; s.rt_pages.clear();
    return false;
}
static uint64_t g_tex_stats[4];   // binds, hits by generation, hits by hash, decodes

static inline uint64_t hash_words(const uint64_t *p, size_t n, uint64_t h) {
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x9E3779B97F4A7C15ull; h ^= h >> 29; }
    return h;
}
// Hash of the source pages: textures are re-streamed through PATH3 every frame, usually with
// identical bytes, so a page write only forces a re-decode when the contents really changed.
static uint64_t hash_pages(const std::vector<uint16_t> &pages) {
    uint64_t h = 0x243F6A8885A308D3ull;
    for (uint16_t p : pages) h = hash_words((const uint64_t *)(g_vram + (size_t)p * 2048), 1024, h ^ p);
    return h;
}
static std::unordered_map<TexKey, TexSlot, TexKeyHash> g_tex_cache;
static uint32_t g_tex_epoch;   // bumped when the cache is cleared (invalidates bound pointers)

static uint32_t page_at(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y) {
    switch (psm) {
    case CT16: return (addr16(bp, bw, x, y, blockTable16) >> 1) >> 11;
    case CT16S: return (addr16(bp, bw, x, y, blockTable16S) >> 1) >> 11;
    case Z16: return (addr16(bp, bw, x, y, blockTableZ16) >> 1) >> 11;
    case Z16S: return (addr16(bp, bw, x, y, blockTableZ16S) >> 1) >> 11;
    case T8: return (addr8(bp, bw, x, y) >> 2) >> 11;
    case T4: return (addr4(bp, bw, x, y) >> 3) >> 11;
    case Z32: case Z24: return addr32(bp, bw, x, y, blockTableZ32) >> 11;
    default: return addr32(bp, bw, x, y, blockTable32) >> 11;
    }
}

static uint32_t page_of(const TexState &t, uint32_t x, uint32_t y) {
    switch (t.psm) {
    case CT16: return (addr16(t.tbp, t.tbw, x, y, blockTable16) >> 1) >> 11;
    case CT16S: return (addr16(t.tbp, t.tbw, x, y, blockTable16S) >> 1) >> 11;
    case T8: return (addr8(t.tbp, t.tbw, x, y) >> 2) >> 11;
    case T4: return (addr4(t.tbp, t.tbw, x, y) >> 3) >> 11;
    default: return addr32(t.tbp, t.tbw, x, y, blockTable32) >> 11;
    }
}

// Debug: when the file dumptex.flag exists, write every decoded texture to texdump/ as BMP.
static void probe_dump();
static void tex_dump_check() {
    static uint32_t n; static ULONGLONG last;
    if (++n % 256) return;
    ULONGLONG now = GetTickCount64();   // poll the flag files at most once a second
    if (now - last < 1000) return;
    last = now;
    static ULONGLONG last_stats;
    if (now - last_stats >= 3000) {
        last_stats = now;
        rt_log(LOG_INFO, "tex: binds %llu gen-hits %llu hash-hits %llu decodes %llu (%zu keys)",
               (unsigned long long)g_tex_stats[0], (unsigned long long)g_tex_stats[1], (unsigned long long)g_tex_stats[2],
               (unsigned long long)g_tex_stats[3], g_tex_cache.size());
        memset(g_tex_stats, 0, sizeof g_tex_stats);
        rt_log(LOG_INFO, "gs wait ms (n): other %llu(%llu) rect %llu(%llu) texpg %llu(%llu) full %llu(%llu) xfer %llu(%llu) display %llu(%llu) war %llu(%llu)",
               g_wait_ns[0] / 1000000, g_wait_n[0], g_wait_ns[1] / 1000000, g_wait_n[1], g_wait_ns[2] / 1000000, g_wait_n[2],
               g_wait_ns[3] / 1000000, g_wait_n[3], g_wait_ns[4] / 1000000, g_wait_n[4], g_wait_ns[5] / 1000000, g_wait_n[5],
               g_wait_ns[6] / 1000000, g_wait_n[6]);
        memset(g_wait_ns, 0, sizeof g_wait_ns); memset(g_wait_n, 0, sizeof g_wait_n);
        pxstat_report();
    }
    if (FILE *pf = fopen("probe.flag", "rb")) { fclose(pf); remove("probe.flag"); probe_dump(); }
    FILE *f = fopen("dumptex.flag", "rb");
    if (!f) return;
    fclose(f); remove("dumptex.flag");
    CreateDirectoryA("texdump", nullptr);
    int k = 0;
    for (auto &slot : g_tex_cache) for (auto &var : slot.second.v) {
        if (var.pix.empty()) continue;
        auto e = std::make_pair(slot.first, &var);
        const TexKey &key = e.first;
        char name[128];
        snprintf(name, sizeof name, "texdump/%04d_tbp%04X_bw%u_psm%02X_%ux%u_cpsm%X_csa%u.bmp", k++, key.tbp, key.tbw, key.psm, key.tw, key.th, key.cpsm, key.csa);
        FILE *o = fopen(name, "wb");
        if (!o) continue;
        uint32_t w = key.tw, h = key.th, sz = 54 + w * h * 4;
        uint8_t hd[54] = {'B', 'M'};
        memcpy(hd + 2, &sz, 4); uint32_t off = 54; memcpy(hd + 10, &off, 4);
        uint32_t hs = 40; memcpy(hd + 14, &hs, 4); memcpy(hd + 18, &w, 4); int32_t nh = -(int32_t)h; memcpy(hd + 22, &nh, 4);
        uint16_t pl = 1, bpp = 32; memcpy(hd + 26, &pl, 2); memcpy(hd + 28, &bpp, 2);
        fwrite(hd, 1, 54, o);
        for (uint32_t i = 0; i < w * h; i++) {
            uint32_t c = e.second->pix[i];
            uint32_t bgra = ((c & 255) << 16) | (c & 0xFF00) | ((c >> 16) & 255) | (c & 0xFF000000u);   // raw alpha (0x80 = 1.0)
            fwrite(&bgra, 4, 1, o);
        }
        fclose(o);
    }
    rt_log(LOG_INFO, "GS: dumped %d textures", k);
}

// Per-primitive fast path: a small direct-mapped memo of recent binds. A hit needs the same
// texture registers, no CLUT load, no cache flush and no write to the texture's pages since.
struct BindMemo {
    uint32_t tbp, tbw, psm, tw, th, cpsm, csa, clut_gen, epoch; uint64_t texa;
    TexEntry *e; TexSlot *slot;
};

// Sample a render target in place: draws queued earlier that write it must finish first, and
// its pages are recorded so later writes wait for the draws reading it.
static void bind_direct(TexState &t, const std::vector<uint16_t> &pages) {
    if (g_wr_since_any) for (uint16_t p : pages) if (g_wr_since[p]) { push_barrier(); break; }
    // Only pages holding GS-drawn data count as read: a render target bound with a generous
    // TW/TH (e.g. 640x224 as 1024x512) also spans unrelated pages that the game keeps uploading
    // textures into, and marking those would make every such upload wait for the rasterizer.
    for (uint16_t p : pages) if (g_page_drawn[p]) { g_rd_cur[p] = 1; g_rd_since[p] = 1; }
    g_rd_any_cur = true; g_rd_since_any = true;
    t.pix = nullptr; t.direct = true;
}
static BindMemo g_bind_memo[16];

static void tex_bind_slow(TexState &t, TexEntry **out, TexSlot **slot_out);
static void tex_bind(TexState &t) {
    BindMemo &m = g_bind_memo[(t.tbp ^ (t.tbp >> 4) ^ t.psm) & 15];
    if (m.slot && m.slot->direct && m.tbp == t.tbp && m.tbw == t.tbw && m.psm == t.psm && m.tw == t.tw && m.th == t.th &&
        m.texa == t.texa && m.epoch == g_tex_epoch && slot_still_direct(*m.slot)) { bind_direct(t, m.slot->rt_pages); return; }
    if (m.e && m.tbp == t.tbp && m.tbw == t.tbw && m.psm == t.psm && m.tw == t.tw && m.th == t.th &&
        m.cpsm == t.cpsm && m.csa == t.csa && m.texa == t.texa && m.clut_gen == g_clut_gen && m.epoch == g_tex_epoch) {
        TexEntry *e = m.e;
        bool ok = true;
        for (uint16_t p : e->pages) if (g_page_gen[p] >= e->gen) { ok = false; break; }
        if (ok) { t.pix = e->pix.data(); t.uid = (uint64_t)(uintptr_t)e ^ e->hash; return; }
    }
    TexEntry *e = nullptr; TexSlot *sl = nullptr;
    tex_bind_slow(t, &e, &sl);
    m = BindMemo{t.tbp, t.tbw, t.psm, t.tw, t.th, t.cpsm, t.csa, g_clut_gen, g_tex_epoch, t.texa, e, sl};
}

static void tex_bind_slow(TexState &t, TexEntry **out, TexSlot **slot_out) {
    tex_dump_check();
    bool indexed = !(t.psm == CT32 || t.psm == CT24 || t.psm == CT16 || t.psm == CT16S);
    bool uses_texa = !indexed ? t.psm != CT32 : t.cpsm != 0;
    // Decoded palette for indexed formats; its hash keys the cache (the CLUT is reloaded per texture).
    struct PalMemo { uint32_t gen = 0, cpsm, csa, is4; uint64_t texa; uint32_t hash; uint32_t pal[256]; };
    static PalMemo pm;
    uint32_t clut_hash = 0;
    if (indexed) {
        uint32_t is4 = t.psm == T4 || t.psm == T4HL || t.psm == T4HH;
        if (pm.gen != g_clut_gen || pm.cpsm != t.cpsm || pm.csa != t.csa || pm.is4 != is4 || (uses_texa && pm.texa != t.texa)) {
            pm.gen = g_clut_gen; pm.cpsm = t.cpsm; pm.csa = t.csa; pm.is4 = is4; pm.texa = t.texa;
            int n = is4 ? 16 : 256;
            for (int i = 0; i < n; i++) {
                Rgba c = clut_lookup(t, (uint32_t)i);
                pm.pal[i] = (uint32_t)c.r | ((uint32_t)c.g << 8) | ((uint32_t)c.b << 16) | ((uint32_t)c.a << 24);
            }
            pm.hash = (uint32_t)hash_words((const uint64_t *)pm.pal, (size_t)n / 2, 0x51ED27ull);
        }
        clut_hash = pm.hash;
    }
    const uint32_t *pal = pm.pal;
    TexKey k{t.tbp, t.tbw, t.psm, t.tw, t.th, indexed ? t.cpsm : 0, indexed ? t.csa : 0, clut_hash,
             uses_texa ? t.texa : 0};
    g_tex_stats[0]++;
    if (g_tex_cache.size() > 2048) { gs_flush(); g_tex_cache.clear(); g_tex_epoch++; memset(g_bind_memo, 0, sizeof g_bind_memo); }
    *slot_out = nullptr;
    TexSlot &slot = g_tex_cache[k];
    *slot_out = &slot;
    slot.tick++;
    if (slot.direct && slot_still_direct(slot)) { bind_direct(t, slot.rt_pages); return; }
    if (slot.cur >= 0) {
        TexEntry &e = slot.v[slot.cur];
        bool ok = true;
        for (uint16_t p : e.pages) if (g_page_gen[p] >= e.gen) { ok = false; break; }
        if (ok) { e.used = slot.tick; t.pix = e.pix.data(); t.uid = (uint64_t)(uintptr_t)&e ^ e.hash; *out = &e; g_tex_stats[1]++; return; }
    }
    if (!indexed) {
        // Drawn by the GS (frame buffer / post-processing target)? Sample it in place from now on.
        std::vector<uint16_t> pg;
        bool drawn = false;
        static uint8_t seen_pg[NPAGES];
        memset(seen_pg, 0, sizeof seen_pg);
        for_rect_pages(t.psm, t.tbp, t.tbw, 0, 0, t.tw, t.th, [&](uint32_t p) {
            if (!seen_pg[p]) { seen_pg[p] = 1; pg.push_back((uint16_t)p); drawn |= g_page_drawn[p] != 0; }
            return false;
        });
        if (drawn) {
            slot.direct = true; slot.rt_pages = std::move(pg); slot.undraw_seen = g_undraw_gen;
            rt_log(LOG_INFO, "GS: sampling render target tbp %04X psm %02X %ux%u in place", t.tbp, t.psm, t.tw, t.th);
            bind_direct(t, slot.rt_pages);
            return;
        }
    }
    int victim = 0;
    if (slot.cur >= 0) {
        // pages changed: see whether the contents match one of the decoded variants
        for (int i = 0; i < 4; i++)
            for (uint16_t p : slot.v[i].pages) if (page_busy(p)) {
                if (rt_once(0xC10000000ull | (t.tbp << 8) | t.psm)) rt_log(LOG_INFO, "texpg wait: tbp %04X psm %02X %ux%u page %u cur %d run %d", t.tbp, t.psm, t.tw, t.th, p, g_pg_cur[p], g_pg_cnt[p]);
                g_wait_reason = 2; gs_flush(); i = 4; break; }   // render target being drawn
        uint64_t h = hash_pages(slot.v[slot.cur].pages);
        for (int i = 0; i < 4; i++) {
            TexEntry &e = slot.v[i];
            if (e.pix.empty()) continue;
            uint64_t hi = (e.pages == slot.v[slot.cur].pages) ? h : hash_pages(e.pages);
            if (hi == e.hash) {
                e.gen = ++g_write_gen; e.used = slot.tick; slot.cur = i;
                t.pix = e.pix.data(); t.uid = (uint64_t)(uintptr_t)&e ^ e.hash; *out = &e; g_tex_stats[2]++; return;
            }
        }
        for (int i = 1; i < 4; i++) if (slot.v[i].used < slot.v[victim].used) victim = i;
    }
    g_tex_stats[3]++;
    slot.cur = victim;
    TexEntry &e = slot.v[victim];
    sync_rect(t.psm, t.tbp, t.tbw, 0, 0, t.tw, t.th, 2);
    if (!e.pix.empty()) g_grave_cur.push_back(std::move(e.pix));   // queued draws may still sample it
    e.pix.clear();
    e.used = slot.tick;
    e.gen = ++g_write_gen;
    e.pix.resize((size_t)t.tw * t.th);
    e.pages.clear();
    static uint8_t seen[VRAM_WORDS / 2048];
    memset(seen, 0, sizeof seen);
    for (uint32_t y = 0; y < t.th; y++) {
        uint32_t *row = &e.pix[(size_t)y * t.tw];
        if (indexed) {
            for (uint32_t x = 0; x < t.tw; x++) row[x] = pal[read_raw(t.psm, t.tbp, t.tbw, x, y) & 255];
        } else if (t.psm == CT32) {
            for (uint32_t x = 0; x < t.tw; x++) row[x] = g_vram[addr32(t.tbp, t.tbw, x, y, blockTable32)];
        } else {
            for (uint32_t x = 0; x < t.tw; x++) {
                Rgba c = texel_slow(t, (int)x, (int)y);
                row[x] = (uint32_t)c.r | ((uint32_t)c.g << 8) | ((uint32_t)c.b << 16) | ((uint32_t)c.a << 24);
            }
        }
        if (!(y & 3))
            for (uint32_t x = 0; x < t.tw; x += 4) {
                uint32_t p = page_of(t, x, y) & (VRAM_WORDS / 2048 - 1);
                if (!seen[p]) { seen[p] = 1; e.pages.push_back((uint16_t)p); }
            }
    }
    e.hash = hash_pages(e.pages);
    t.pix = e.pix.data();
    t.uid = (uint64_t)(uintptr_t)&e ^ e.hash;
    *out = &e;
}

static inline int wrap_coord(int c, uint32_t mode, uint32_t size, uint32_t mn, uint32_t mx) {
    switch (mode) {
    case 0: return c & (int)(size - 1);
    case 1: return c < 0 ? 0 : c >= (int)size ? (int)size - 1 : c;
    case 2: c = c < (int)mn ? (int)mn : c > (int)mx ? (int)mx : c; return c & (int)(size - 1);
    default: return (int)(((uint32_t)c & mn) | mx) & (int)(size - 1);
    }
}

// One mip level: its texels, size and the CLAMP region scaled to it.
struct TexLevel { const uint32_t *pix; uint32_t tw, th, wms, wmt, minu, maxu, minv, maxv; uint32_t psm, tbp, tbw; uint64_t texa; };

static uint32_t direct_texel(const TexLevel &t, int u, int v) {
    bool aem = bits(t.texa, 15, 1) != 0;
    switch (t.psm) {
    case CT32: return g_vram[addr32(t.tbp, t.tbw, (uint32_t)u, (uint32_t)v, blockTable32)];
    case CT24: {
        uint32_t c = g_vram[addr32(t.tbp, t.tbw, (uint32_t)u, (uint32_t)v, blockTable32)] & 0xFFFFFF;
        return c | ((aem && !c) ? 0u : (uint32_t)bits(t.texa, 0, 8) << 24);
    }
    default: {
        Rgba c = from16(read_raw(t.psm, t.tbp, t.tbw, (uint32_t)u, (uint32_t)v), t.texa, aem);
        return (uint32_t)c.r | ((uint32_t)c.g << 8) | ((uint32_t)c.b << 16) | ((uint32_t)c.a << 24);
    }
    }
}

static inline uint32_t texel_packed(const TexLevel &t, int u, int v) {
    u = wrap_coord(u, t.wms, t.tw, t.minu, t.maxu);
    v = wrap_coord(v, t.wmt, t.th, t.minv, t.maxv);
    if (!t.pix) return direct_texel(t, u, v);
    return t.pix[(size_t)v * t.tw + u];
}

static inline Rgba sample_level(const TexLevel &t, float u, float v, bool bilinear) {
    if (!bilinear) return from32(texel_packed(t, (int)std::floor(u), (int)std::floor(v)));
    u -= 0.5f; v -= 0.5f;
    float fu0 = std::floor(u), fv0 = std::floor(v);
    int u0 = (int)fu0, v0 = (int)fv0;
    int fu = (int)((u - fu0) * 256), fv = (int)((v - fv0) * 256);
    uint32_t a = texel_packed(t, u0, v0), b = texel_packed(t, u0 + 1, v0), c = texel_packed(t, u0, v0 + 1), d = texel_packed(t, u0 + 1, v0 + 1);
    auto mix = [&](int sh) {
        int p = (a >> sh) & 255, q = (b >> sh) & 255, r = (c >> sh) & 255, s2 = (d >> sh) & 255;
        int top = p * (256 - fu) + q * fu, bot = r * (256 - fu) + s2 * fu;
        return (top * (256 - fv) + bot * fv + 32768) >> 16;
    };
    return Rgba{mix(0), mix(8), mix(16), mix(24)};
}

static inline float fast_log2(float x) {
    uint32_t b; memcpy(&b, &x, 4);
    return (float)((int)((b >> 23) & 255) - 127) + (float)(b & 0x7FFFFF) * (1.0f / 8388608.0f);
}

// u, v are in level-0 texels; q is the interpolated Q (for the LOD).
static inline Rgba sample(const TexState &t, float u, float v, float q) {
    int lev = 0;
    bool bilinear = t.bilinear;
    if (t.nlev > 1) {
        float lod = t.lcm ? t.k : -fast_log2(std::fabs(q) + 1e-30f) * (float)(1 << t.lshift) + t.k;
        if (lod > 0) {
            bilinear = t.min_linear;
            lev = std::min((int)(lod + 0.5f), t.nlev - 1);
        }
    }
    if (lev == 0) {
        TexLevel l{t.pix, t.tw, t.th, t.wms, t.wmt, t.minu, t.maxu, t.minv, t.maxv, t.psm, t.tbp, t.tbw, t.texa};
        return sample_level(l, u, v, bilinear);
    }
    float sc = 1.0f / (float)(1 << lev);
    TexLevel l{t.lpix[lev], std::max(t.tw >> lev, 1u), std::max(t.th >> lev, 1u), t.wms, t.wmt,
               t.minu >> lev, t.maxu >> lev, t.minv >> lev, t.maxv >> lev, t.psm, t.mtbp[lev], t.mtbw[lev], t.texa};
    if (t.wms == 3) { l.minu = t.minu; l.maxu = t.maxu >> lev; }   // REGION_REPEAT masks: keep the mask, scale the offset
    if (t.wmt == 3) { l.minv = t.minv; l.maxv = t.maxv >> lev; }
    return sample_level(l, u * sc, v * sc, bilinear);
}

// ================================================================ drawing
struct Vertex { float x, y; uint32_t z; int r, g, b, a; float s, t, q, u, v; int fog; };

struct DrawCtx {
    uint32_t prim;
    bool tme, fge, abe, fst, iip;
    int ctxt;
    // frame / z
    uint32_t fbp, fbw, fpsm, fbmsk;
    uint32_t zbp, zpsm; bool zmsk;
    // test
    bool ate; uint32_t atst, aref, afail; bool date, datm; bool zte; uint32_t ztst;
    // alpha
    uint32_t aa, ab, ac, ad, afix; bool pabe, fba, colclamp;
    int scx0, scx1, scy0, scy1;
    TexState tex;
    int fogr, fogg, fogb;
    uint32_t vu1start;
};

// debug: pixel history. NPPL_PROBE="x,y" records the draws touching that pixel; touch
// probe.flag in the working directory to log the most recent ones.
static int g_probe_x = -1, g_probe_y = -1;
static thread_local const Vertex *t_cur_v;
struct ProbeRec { DrawCtx d; Vertex v[3]; uint32_t z, zd; Rgba src; uint32_t dst, out; int stage; uint32_t vu1; };
extern uint32_t g_vu1_last_start;
static std::mutex g_probe_mx;
static std::vector<ProbeRec> g_probe;
static void probe_rec(const DrawCtx &d, uint32_t z, uint32_t zd, Rgba src, uint32_t dst, uint32_t out, int stage) {
    std::lock_guard<std::mutex> lk(g_probe_mx);
    ProbeRec r{d, {}, z, zd, src, dst, out, stage, d.vu1start};
    if (t_cur_v) for (int i = 0; i < 3; i++) r.v[i] = t_cur_v[i];
    g_probe.push_back(r);
    if (g_probe.size() > 400) g_probe.erase(g_probe.begin(), g_probe.begin() + 100);
}
static void probe_dump() {
    std::lock_guard<std::mutex> lk(g_probe_mx);
    for (auto &r : g_probe) {
        const DrawCtx &d = r.d;
        rt_log(LOG_INFO, "PROBE vu%04X st%d prim%u tme%d tbp%04X psm%02X tfx%u abe%d a(%u%u%u%u fix%u) ate%d atst%u aref%u afail%u zte%d ztst%u zmsk%d fbp%04X fbmsk%08X fge%d | z %08X zd %08X src %d,%d,%d,%d dst %08X out %08X",
               r.vu1, r.stage, d.prim, d.tme, d.tex.tbp, d.tex.psm, d.tex.tfx, d.abe, d.aa, d.ab, d.ac, d.ad, d.afix, d.ate, d.atst, d.aref, d.afail, d.zte, d.ztst, d.zmsk, d.fbp, d.fbmsk, d.fge,
               r.z, r.zd, r.src.r, r.src.g, r.src.b, r.src.a, r.dst, r.out);
        for (int i = 0; i < 3; i++) rt_log(LOG_INFO, "   v%d %.1f,%.1f z%08X rgba %d,%d,%d,%d stq %g,%g,%g fog %d", i, r.v[i].x, r.v[i].y, r.v[i].z, r.v[i].r, r.v[i].g, r.v[i].b, r.v[i].a, r.v[i].s, r.v[i].t, r.v[i].q, r.v[i].fog);
    }
    g_probe.clear();
}

static Vertex g_vq[3];
static int g_vcount;
static uint32_t g_draw_count;

static void build_ctx(DrawCtx &d) {
    d.vu1start = g_vu1_last_start;
    uint64_t prim = g_reg[R_PRIM];
    uint64_t attrs = (g_reg[R_PRMODECONT] & 1) ? prim : g_reg[R_PRMODE];
    d.prim = (uint32_t)(prim & 7);
    d.iip = bits(attrs, 3, 1); d.tme = bits(attrs, 4, 1); d.fge = bits(attrs, 5, 1); d.abe = bits(attrs, 6, 1);
    d.fst = bits(attrs, 8, 1); d.ctxt = (int)bits(attrs, 9, 1);
    int k = d.ctxt;
    uint64_t frame = g_reg[R_FRAME_1 + k], zbuf = g_reg[R_ZBUF_1 + k], test = g_reg[R_TEST_1 + k];
    uint64_t alpha = g_reg[R_ALPHA_1 + k], sc = g_reg[R_SCISSOR_1 + k], tex0 = g_reg[R_TEX0_1 + k];
    uint64_t clamp = g_reg[R_CLAMP_1 + k], tex1 = g_reg[R_TEX1_1 + k];
    d.fbp = (uint32_t)bits(frame, 0, 9) * 32; d.fbw = (uint32_t)bits(frame, 16, 6); d.fpsm = (uint32_t)bits(frame, 24, 6);
    d.fbmsk = (uint32_t)(frame >> 32);
    d.zbp = (uint32_t)bits(zbuf, 0, 9) * 32; d.zpsm = 0x30 | (uint32_t)bits(zbuf, 24, 4); d.zmsk = bits(zbuf, 32, 1);
    d.ate = bits(test, 0, 1); d.atst = (uint32_t)bits(test, 1, 3); d.aref = (uint32_t)bits(test, 4, 8);
    d.afail = (uint32_t)bits(test, 12, 2); d.date = bits(test, 14, 1); d.datm = bits(test, 15, 1);
    d.zte = bits(test, 16, 1); d.ztst = (uint32_t)bits(test, 17, 2);
    d.aa = (uint32_t)bits(alpha, 0, 2); d.ab = (uint32_t)bits(alpha, 2, 2); d.ac = (uint32_t)bits(alpha, 4, 2);
    d.ad = (uint32_t)bits(alpha, 6, 2); d.afix = (uint32_t)bits(alpha, 32, 8);
    d.pabe = g_reg[R_PABE] & 1; d.fba = g_reg[R_FBA_1 + k] & 1; d.colclamp = g_reg[R_COLCLAMP] & 1;
    d.scx0 = (int)bits(sc, 0, 11); d.scx1 = (int)bits(sc, 16, 11); d.scy0 = (int)bits(sc, 32, 11); d.scy1 = (int)bits(sc, 48, 11);
    TexState &t = d.tex;
    t.tbp = (uint32_t)bits(tex0, 0, 14); t.tbw = (uint32_t)bits(tex0, 14, 6); t.psm = (uint32_t)bits(tex0, 20, 6);
    t.tw = 1u << std::min<uint32_t>((uint32_t)bits(tex0, 26, 4), 10); t.th = 1u << std::min<uint32_t>((uint32_t)bits(tex0, 30, 4), 10);
    t.tcc = (uint32_t)bits(tex0, 34, 1); t.tfx = (uint32_t)bits(tex0, 35, 2);
    t.cpsm = (uint32_t)bits(tex0, 51, 4); t.csa = (uint32_t)bits(tex0, 56, 5);
    t.wms = (uint32_t)bits(clamp, 0, 2); t.wmt = (uint32_t)bits(clamp, 2, 2);
    t.minu = (uint32_t)bits(clamp, 4, 10); t.maxu = (uint32_t)bits(clamp, 14, 10);
    t.minv = (uint32_t)bits(clamp, 24, 10); t.maxv = (uint32_t)bits(clamp, 34, 10);
    t.texa = g_reg[R_TEXA];
    t.bilinear = bits(tex1, 5, 1) != 0;
    t.pix = nullptr; t.direct = false;
    {
        uint32_t mxl = (uint32_t)bits(tex1, 2, 3), mmin = (uint32_t)bits(tex1, 6, 3);
        t.nlev = (mmin >= 2 && mmin <= 5) ? 1 + (int)std::min(mxl, 6u) : 1;
        t.min_linear = mmin == 1 || mmin == 4 || mmin == 5;
        t.lcm = bits(tex1, 0, 1) != 0;
        t.lshift = (int)bits(tex1, 19, 2);
        t.k = (float)((int32_t)(bits(tex1, 32, 12) << 20) >> 20) / 16.0f;
        uint64_t m1 = g_reg[R_MIPTBP1_1 + k], m2 = g_reg[R_MIPTBP2_1 + k];
        t.mtbp[0] = t.tbp; t.mtbw[0] = t.tbw;
        for (int i = 0; i < 3; i++) {
            t.mtbp[1 + i] = (uint32_t)bits(m1, i * 20, 14); t.mtbw[1 + i] = (uint32_t)bits(m1, i * 20 + 14, 6);
            t.mtbp[4 + i] = (uint32_t)bits(m2, i * 20, 14); t.mtbw[4 + i] = (uint32_t)bits(m2, i * 20 + 14, 6);
        }
        for (int i = 0; i < 7; i++) t.lpix[i] = nullptr;
    }
    uint64_t fc = g_reg[R_FOGCOL];
    d.fogr = (int)bits(fc, 0, 8); d.fogg = (int)bits(fc, 8, 8); d.fogb = (int)bits(fc, 16, 8);
}

static inline uint32_t zmax(uint32_t zpsm) { return zpsm == Z32 ? 0xFFFFFFFFu : zpsm == Z24 ? 0xFFFFFF : 0xFFFF; }

uint64_t g_prof_pixels, g_prof_prims[8];
#include <atomic>
static std::atomic<uint64_t> g_px_total;
static thread_local uint64_t t_px;
// Rasterizer threads own interleaved 16-row bands: thread k draws rows with (y / 16) % t_row_n == k.
// Bands (rather than single rows) let small primitives skip the threads they don't touch.
static thread_local int t_row_k = 0, t_row_n = 1;
static int g_band_sh = 4, g_band_mask = 15;   // band height 2^g_band_sh rows (NPPL_BAND)
static inline bool my_row(int y) { return t_row_n == 1 || ((y >> g_band_sh) % t_row_n) == t_row_k; }
// First row >= y0 owned by this thread.
static inline int my_first_row(int y0) {
    if (t_row_n == 1) return y0;
    int band = y0 >> g_band_sh, off = ((t_row_k - band) % t_row_n + t_row_n) % t_row_n;
    return off == 0 ? y0 : (band + off) << g_band_sh;
}

// Address of a frame/z buffer pixel: word index (32-bit formats) or halfword index (16-bit).
static inline uint32_t fb_addr(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y) {
    switch (psm) {
    case CT32: case CT24: return addr32(bp, bw, x, y, blockTable32);
    case Z32: case Z24: return addr32(bp, bw, x, y, blockTableZ32);
    case CT16: return addr16(bp, bw, x, y, blockTable16);
    case CT16S: return addr16(bp, bw, x, y, blockTable16S);
    case Z16: return addr16(bp, bw, x, y, blockTableZ16);
    default: return addr16(bp, bw, x, y, blockTableZ16S);
    }
}

static void put_pixel(const DrawCtx &d, int x, int y, uint32_t z, Rgba c, int fog) {
    (void)fog;
    t_px++;
    if (x < d.scx0 || x > d.scx1 || y < d.scy0 || y > d.scy1) return;
    const bool probe = x == g_probe_x && y == g_probe_y;
    bool fb_write = true, z_write = !d.zmsk, a_write = true;
    if (d.ate) {
        bool pass;
        int a = c.a, ref = (int)d.aref;
        switch (d.atst) {
        case 0: pass = false; break;
        case 1: pass = true; break;
        case 2: pass = a < ref; break;
        case 3: pass = a <= ref; break;
        case 4: pass = a == ref; break;
        case 5: pass = a >= ref; break;
        case 6: pass = a > ref; break;
        default: pass = a != ref; break;
        }
        if (!pass) {
            switch (d.afail) {
            case 0: return;
            case 1: z_write = false; break;
            case 2: fb_write = false; break;
            case 3: z_write = false; a_write = false; break;
            }
        }
    }
    bool is16 = d.fpsm == CT16 || d.fpsm == CT16S;
    uint32_t fa = fb_addr(d.fpsm, d.fbp, d.fbw, (uint32_t)x, (uint32_t)y);
    uint32_t dst = is16 ? vram16()[fa] : g_vram[fa];
    if (d.date) {
        uint32_t abit = is16 ? (dst >> 15) & 1 : (dst >> 31) & 1;
        if (abit != (uint32_t)d.datm) return;
    }
    uint32_t za = 0;
    bool z16 = d.zpsm == Z16 || d.zpsm == Z16S;
    if (d.zte) {
        uint32_t zm = zmax(d.zpsm);
        if (z > zm) z = zm;
        za = fb_addr(d.zpsm, d.zbp, d.fbw, (uint32_t)x, (uint32_t)y);
        uint32_t zd = (z16 ? vram16()[za] : g_vram[za]) & zm;
        bool zfail = d.ztst == 0 || (d.ztst == 2 && z < zd) || (d.ztst == 3 && z <= zd);
        if (probe) probe_rec(d, z, zd, c, dst, 0, zfail ? 1 : 0);
        if (zfail) return;
    } else z_write = false;
    if (fb_write) {
        int dr, dg, db, da;
        if (is16) { dr = (dst & 31) << 3; dg = ((dst >> 5) & 31) << 3; db = ((dst >> 10) & 31) << 3; da = (dst & 0x8000) ? 0x80 : 0; }
        else { dr = dst & 255; dg = (dst >> 8) & 255; db = (dst >> 16) & 255; da = dst >> 24; }
        if (d.abe && (!d.pabe || (c.a & 0x80))) {
            int ca = d.ac == 0 ? c.a : d.ac == 1 ? da : (int)d.afix;
            int sr[3] = {c.r, dr, 0}, sg[3] = {c.g, dg, 0}, sb[3] = {c.b, db, 0};
            int vr = (((sr[d.aa] - sr[d.ab]) * ca) >> 7) + sr[d.ad];
            int vg = (((sg[d.aa] - sg[d.ab]) * ca) >> 7) + sg[d.ad];
            int vb = (((sb[d.aa] - sb[d.ab]) * ca) >> 7) + sb[d.ad];
            if (d.colclamp) { c.r = std::clamp(vr, 0, 255); c.g = std::clamp(vg, 0, 255); c.b = std::clamp(vb, 0, 255); }
            else { c.r = vr & 255; c.g = vg & 255; c.b = vb & 255; }
        } else if (d.colclamp) {
            c.r = std::clamp(c.r, 0, 255); c.g = std::clamp(c.g, 0, 255); c.b = std::clamp(c.b, 0, 255);
        } else { c.r &= 255; c.g &= 255; c.b &= 255; }
        int a = std::clamp(c.a, 0, 255);
        if (d.fba) a |= 0x80;
        uint32_t mask = d.fbmsk;
        if (is16) {
            uint32_t out = (uint32_t)(c.r >> 3) | ((uint32_t)(c.g >> 3) << 5) | ((uint32_t)(c.b >> 3) << 10) | ((a & 0x80) ? 0x8000u : 0);
            uint32_t m16 = ((mask >> 3) & 0x1F) | ((mask >> 6) & 0x3E0) | ((mask >> 9) & 0x7C00) | ((mask >> 16) & 0x8000);
            if (!a_write) m16 |= 0x8000;
            vram16()[fa] = (uint16_t)((out & ~m16) | (dst & m16));
        } else {
            uint32_t out = (uint32_t)c.r | ((uint32_t)c.g << 8) | ((uint32_t)c.b << 16) | ((uint32_t)a << 24);
            if (!a_write || d.fpsm == CT24) mask |= 0xFF000000u;
            g_vram[fa] = (out & ~mask) | (dst & mask);
            if (probe) probe_rec(d, z, 0, c, dst, g_vram[fa], 2);
        }
    }
    if (z_write) {
        if (z16) vram16()[za] = (uint16_t)z;
        else if (d.zpsm == Z24) g_vram[za] = (g_vram[za] & 0xFF000000u) | (z & 0xFFFFFF);
        else g_vram[za] = z;
    }
}

static Rgba shade(const DrawCtx &d, Rgba vc, float u, float v, int fog, float q = 1.0f) {
    Rgba c = vc;
    if (d.tme) {
        Rgba t = sample(d.tex, u, v, q);
        switch (d.tex.tfx) {
        case 0:
            c.r = (t.r * vc.r) >> 7; c.g = (t.g * vc.g) >> 7; c.b = (t.b * vc.b) >> 7;
            c.a = d.tex.tcc ? (t.a * vc.a) >> 7 : vc.a;
            break;
        case 1: c = t; if (!d.tex.tcc) c.a = vc.a; break;
        case 2:
            c.r = ((t.r * vc.r) >> 7) + vc.a; c.g = ((t.g * vc.g) >> 7) + vc.a; c.b = ((t.b * vc.b) >> 7) + vc.a;
            c.a = d.tex.tcc ? t.a + vc.a : vc.a;
            break;
        default:
            c.r = ((t.r * vc.r) >> 7) + vc.a; c.g = ((t.g * vc.g) >> 7) + vc.a; c.b = ((t.b * vc.b) >> 7) + vc.a;
            c.a = d.tex.tcc ? t.a : vc.a;
            break;
        }
        c.r = std::min(c.r, 255); c.g = std::min(c.g, 255); c.b = std::min(c.b, 255); c.a = std::min(c.a, 255);
    }
    if (d.fge) {
        c.r = (fog * c.r + (255 - fog) * d.fogr) >> 8;
        c.g = (fog * c.g + (255 - fog) * d.fogg) >> 8;
        c.b = (fog * c.b + (255 - fog) * d.fogb) >> 8;
    }
    return c;
}

static void tex_coords(const DrawCtx &d, float s, float t, float q, float u, float v, float &tu, float &tv) {
    if (d.fst) { tu = u; tv = v; return; }
    if (q == 0) q = 1e-6f;
    tu = s / q * (float)d.tex.tw; tv = t / q * (float)d.tex.th;
}

static void draw_sprite(const DrawCtx &d, const Vertex &a, const Vertex &b) {
    const Vertex &l = a.x <= b.x ? a : b, &r = a.x <= b.x ? b : a;
    const Vertex &tp = a.y <= b.y ? a : b, &bt = a.y <= b.y ? b : a;
    int ix0 = std::max((int)std::ceil(l.x), d.scx0), ix1 = std::min((int)std::ceil(r.x), d.scx1 + 1);
    int iy0 = std::max((int)std::ceil(tp.y), d.scy0), iy1 = std::min((int)std::ceil(bt.y), d.scy1 + 1);
    if (ix0 >= ix1 || iy0 >= iy1) return;
    float dx = r.x - l.x, dy = bt.y - tp.y;
    Rgba vc{b.r, b.g, b.b, b.a};
    // texture coordinate per pixel step
    float u0, du, v0, dv;
    if (d.fst) {
        du = dx > 0 ? (r.u - l.u) / dx : 0; u0 = l.u + (ix0 - l.x) * du;
        dv = dy > 0 ? (bt.v - tp.v) / dy : 0; v0 = tp.v + (iy0 - tp.y) * dv;
    } else {
        float q = b.q != 0 ? b.q : 1e-6f;
        float su = (float)d.tex.tw / q, sv = (float)d.tex.th / q;
        du = dx > 0 ? (r.s - l.s) / dx * su : 0; u0 = l.s * su + (ix0 - l.x) * du;
        dv = dy > 0 ? (bt.t - tp.t) / dy * sv : 0; v0 = tp.t * sv + (iy0 - tp.y) * dv;
    }
    if (!d.tme && !d.abe && (!d.ate || d.atst == 1) && !d.date && (!d.zte || d.ztst <= 1) && g_probe_x < 0) {
        // Flat fill (clears, Z-only clears): no per-pixel tests, just masked writes.
        if (d.zte && d.ztst == 0) return;
        Rgba c = shade(d, vc, 0, 0, b.fog);
        int a = std::clamp(c.a, 0, 255);
        if (d.fba) a |= 0x80;
        int cr = std::clamp(c.r, 0, 255), cg = std::clamp(c.g, 0, 255), cb = std::clamp(c.b, 0, 255);
        if (!d.colclamp) { cr = c.r & 255; cg = c.g & 255; cb = c.b & 255; }
        bool is16 = d.fpsm == CT16 || d.fpsm == CT16S;
        uint32_t mask = d.fbmsk;
        if (d.fpsm == CT24) mask |= 0xFF000000u;
        uint32_t out32 = (uint32_t)cr | ((uint32_t)cg << 8) | ((uint32_t)cb << 16) | ((uint32_t)a << 24);
        uint32_t out16 = (uint32_t)(cr >> 3) | ((uint32_t)(cg >> 3) << 5) | ((uint32_t)(cb >> 3) << 10) | ((a & 0x80) ? 0x8000u : 0);
        uint32_t m16 = ((mask >> 3) & 0x1F) | ((mask >> 6) & 0x3E0) | ((mask >> 9) & 0x7C00) | ((mask >> 16) & 0x8000);
        bool fbw = is16 ? m16 != 0xFFFF : mask != 0xFFFFFFFFu;
        bool zw = d.zte && !d.zmsk;
        uint32_t zm = zmax(d.zpsm), z = std::min(b.z, zm);
        bool z16 = d.zpsm == Z16 || d.zpsm == Z16S;
        int x0 = std::max(ix0, d.scx0), x1 = std::min(ix1 - 1, d.scx1);
        int y0 = std::max(iy0, d.scy0), y1 = std::min(iy1 - 1, d.scy1);
        for (int y = my_first_row(y0); y <= y1; y = ((y & g_band_mask) == g_band_mask) ? my_first_row(y + 1) : y + 1) {
            t_px += (uint64_t)std::max(0, x1 - x0 + 1);
            for (int x = x0; x <= x1; x++) {
                if (fbw) {
                    uint32_t fa = fb_addr(d.fpsm, d.fbp, d.fbw, (uint32_t)x, (uint32_t)y);
                    if (is16) vram16()[fa] = (uint16_t)((out16 & ~m16) | (vram16()[fa] & m16));
                    else g_vram[fa] = (out32 & ~mask) | (g_vram[fa] & mask);
                }
                if (zw) {
                    uint32_t za = fb_addr(d.zpsm, d.zbp, d.fbw, (uint32_t)x, (uint32_t)y);
                    if (z16) vram16()[za] = (uint16_t)z;
                    else if (d.zpsm == Z24) g_vram[za] = (g_vram[za] & 0xFF000000u) | (z & 0xFFFFFF);
                    else g_vram[za] = z;
                }
            }
        }
        return;
    }
    bool simple = !d.tme && !d.fge;
    Rgba flat = simple ? shade(d, vc, 0, 0, b.fog) : vc;
    float v = v0;
    for (int y = iy0; y < iy1; y++, v += dv) {
        if (!my_row(y)) continue;
        float u = u0;
        for (int x = ix0; x < ix1; x++, u += du)
            put_pixel(d, x, y, b.z, simple ? flat : shade(d, vc, u, v, b.fog, b.q), b.fog);
    }
}

// Scanline triangle rasterizer: all attributes are planes a*x + b*y + c, stepped per pixel.
struct Plane { float dx, dy, c; float at(float x, float y) const { return dx * x + dy * y + c; } };

static void draw_triangle(const DrawCtx &d, const Vertex &v0, const Vertex &v1, const Vertex &v2) {
    Vertex pv[3] = {v0, v1, v2}; t_cur_v = pv;
    struct ClearCur { ~ClearCur() { t_cur_v = nullptr; } } clear_cur;
    float area = (v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y);
    if (g_probe_x >= 0 && my_row(g_probe_y) && std::fabs(area) > 0.5f) {
        float px = (float)g_probe_x, py = (float)g_probe_y;
        auto ed = [&](const Vertex &a, const Vertex &b) { return (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x); };
        float e0 = ed(v0, v1), e1 = ed(v1, v2), e2 = ed(v2, v0);
        if ((e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0)) probe_rec(d, 0, 0, Rgba{}, 0, 0, 3);
    }
    if (area == 0 || !std::isfinite(area)) return;
    int miny = std::max((int)std::ceil(std::min({v0.y, v1.y, v2.y})), d.scy0);
    int maxy = std::min((int)std::ceil(std::max({v0.y, v1.y, v2.y})) - 1, d.scy1);
    if (miny > maxy) return;
    if (my_first_row(miny) > maxy) return;   // no band of this thread
    float inv = 1.0f / area;
    auto plane = [&](float a0, float a1, float a2) {
        Plane p;
        p.dx = ((a1 - a0) * (v2.y - v0.y) - (a2 - a0) * (v1.y - v0.y)) * inv;
        p.dy = ((a2 - a0) * (v1.x - v0.x) - (a1 - a0) * (v2.x - v0.x)) * inv;
        p.c = a0 - p.dx * v0.x - p.dy * v0.y;
        return p;
    };
    bool gour = d.iip;
    Plane pr{}, pg{}, pb{}, pa{}, ps{}, pt{}, pq{}, pf{};
    if (gour) { pr = plane((float)v0.r, (float)v1.r, (float)v2.r); pg = plane((float)v0.g, (float)v1.g, (float)v2.g);
                pb = plane((float)v0.b, (float)v1.b, (float)v2.b); pa = plane((float)v0.a, (float)v1.a, (float)v2.a); }
    // z in double precision (32-bit depth)
    double zdx, zdy, zc;
    {
        double inv_d = 1.0 / (double)area;
        double a0 = v0.z, a1 = v1.z, a2 = v2.z;
        zdx = ((a1 - a0) * (v2.y - v0.y) - (a2 - a0) * (v1.y - v0.y)) * inv_d;
        zdy = ((a2 - a0) * (v1.x - v0.x) - (a1 - a0) * (v2.x - v0.x)) * inv_d;
        zc = a0 - zdx * v0.x - zdy * v0.y;
    }
    if (d.tme) {
        if (d.fst) { ps = plane(v0.u, v1.u, v2.u); pt = plane(v0.v, v1.v, v2.v); }
        else { ps = plane(v0.s, v1.s, v2.s); pt = plane(v0.t, v1.t, v2.t); pq = plane(v0.q, v1.q, v2.q); }
    }
    if (d.fge) pf = plane((float)v0.fog, (float)v1.fog, (float)v2.fog);
    Rgba flat{v2.r, v2.g, v2.b, v2.a};
    const Vertex *vs[3] = {&v0, &v1, &v2};
    for (int y = my_first_row(miny); y <= maxy; y = ((y & g_band_mask) == g_band_mask) ? my_first_row(y + 1) : y + 1) {
        float fy = (float)y;
        // span: intersect the scanline with the three edges
        float xl = -1e30f, xr = 1e30f;
        bool empty = false;
        for (int e = 0; e < 3; e++) {
            const Vertex &a = *vs[(e + 1) % 3], &b = *vs[(e + 2) % 3];   // edge opposite vertex e
            // inside: sign(area) * ((b.x-a.x)*(y-a.y) - (b.y-a.y)*(x-a.x)) >= 0
            float A = -(b.y - a.y), C = (b.x - a.x) * (fy - a.y);
            if (area < 0) { A = -A; C = -C; }
            // A*(x - a.x) + C >= 0
            if (A == 0) { if (C < 0) empty = true; continue; }
            float xb = a.x - C / A;
            if (A > 0) xl = std::max(xl, xb); else xr = std::min(xr, xb);
        }
        if (empty) continue;
        int x0 = std::max((int)std::ceil(xl), d.scx0), x1 = std::min((int)std::ceil(xr) - 1, d.scx1);
        if (x0 > x1) continue;
        float fx = (float)x0;
        float r = pr.at(fx, fy), g = pg.at(fx, fy), bb = pb.at(fx, fy), al = pa.at(fx, fy);
        float s0 = ps.at(fx, fy), t0 = pt.at(fx, fy), q0 = pq.at(fx, fy), f0 = pf.at(fx, fy);
        double z = zdx * x0 + zdy * fy + zc;
        for (int x = x0; x <= x1; x++) {
            Rgba vc = gour ? Rgba{std::clamp((int)r, 0, 255), std::clamp((int)g, 0, 255), std::clamp((int)bb, 0, 255), std::clamp((int)al, 0, 255)} : flat;
            float u = 0, v = 0;
            if (d.tme) {
                if (d.fst) { u = s0; v = t0; }
                else { float q = q0 != 0 ? q0 : 1e-6f; u = s0 / q * (float)d.tex.tw; v = t0 / q * (float)d.tex.th; }
            }
            int fog = d.fge ? std::clamp((int)f0, 0, 255) : 255;
            uint32_t zi = z <= 0 ? 0 : z >= 4294967295.0 ? 0xFFFFFFFFu : (uint32_t)z;
            put_pixel(d, x, y, zi, shade(d, vc, u, v, fog, d.fst ? v2.q : q0), fog);
            r += pr.dx; g += pg.dx; bb += pb.dx; al += pa.dx;
            s0 += ps.dx; t0 += pt.dx; q0 += pq.dx; f0 += pf.dx; z += zdx;
        }
    }
}

static void draw_line(const DrawCtx &d, const Vertex &a, const Vertex &b) {
    float dx = b.x - a.x, dy = b.y - a.y;
    int n = (int)std::max(std::fabs(dx), std::fabs(dy));
    if (n == 0) n = 1;
    for (int i = 0; i <= n; i++) {
        float f = (float)i / n;
        Rgba vc = d.iip ? Rgba{(int)(a.r + (b.r - a.r) * f), (int)(a.g + (b.g - a.g) * f), (int)(a.b + (b.b - a.b) * f), (int)(a.a + (b.a - a.a) * f)}
                        : Rgba{b.r, b.g, b.b, b.a};
        float u = 0, v = 0;
        if (d.tme) tex_coords(d, a.s + (b.s - a.s) * f, a.t + (b.t - a.t) * f, a.q + (b.q - a.q) * f,
                              a.u + (b.u - a.u) * f, a.v + (b.v - a.v) * f, u, v);
        uint32_t z = (uint32_t)((double)a.z + ((double)b.z - (double)a.z) * f);
        if (!my_row((int)std::floor(a.y + dy * f))) continue;
        put_pixel(d, (int)std::floor(a.x + dx * f), (int)std::floor(a.y + dy * f), z, shade(d, vc, u, v, b.fog, a.q + (b.q - a.q) * f), b.fog);
    }
}

// ================================================================ parallel draw queue
struct DrawCmd { DrawCtx d; Vertex v[3]; };
static std::vector<DrawCmd> g_cmds;
static int g_nworkers;
static std::mutex g_pool_mx;
static std::condition_variable g_pool_cv, g_done_cv;

// Rasterizer threads meet here before running anything queued after a barrier command.
struct SpinBarrier {
    std::atomic<int> count{0}, gen{0};
    int n = 1;
    void wait() {
        int g = gen.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
            count.store(0, std::memory_order_relaxed);
            gen.fetch_add(1, std::memory_order_release);
            return;
        }
        for (int i = 0; gen.load(std::memory_order_acquire) == g; i++) {
            if (i < 4000) _mm_pause(); else SwitchToThread();
        }
    }
};
static SpinBarrier g_barrier;
static std::atomic<uint64_t> g_barrier_ns, g_barrier_n;
enum { PRIM_BARRIER = 7 };

static void push_barrier() {
    DrawCmd cmd{};
    cmd.d.prim = PRIM_BARRIER;
    g_cmds.push_back(cmd);
    if (g_wr_since_any) { memset(g_wr_since, 0, sizeof g_wr_since); g_wr_since_any = false; }
    if (g_rd_since_any) { memset(g_rd_since, 0, sizeof g_rd_since); g_rd_since_any = false; }
}

// ================================================================ specialised rasterizer
// Fast versions of draw_triangle / draw_sprite for the common cases: every per-draw choice that
// put_pixel / shade / sample make per pixel (texture on/off, filter, fog, Z test, blending,
// 16/32-bit frame buffer) becomes a template parameter, and frame/Z addresses come from per-row
// swizzle tables instead of the block/column formula. The arithmetic mirrors the generic path
// step for step, so the output is identical (NPPL_RASTCHECK=<n> compares every n-th draw).
// Not covered (generic path): mipmapping, frame formats other than CT32/24/16/16S, probes.

// One frame/Z buffer row: address of pixel x = (base + (x >> 6) * pgsz + sw[x & 63]) & mask.
struct FbRow { uint32_t base; const uint16_t *sw; uint32_t pgsz, mask; };
static inline FbRow fb_row(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t y) {
    bw = std::max(bw, 1u);
    switch (psm) {
    case CT32: case CT24: return {bp * 64 + (y >> 5) * bw * 2048, g_swz32[0][y & 31], 2048, VRAM_WORDS - 1};
    case Z32: case Z24: return {bp * 64 + (y >> 5) * bw * 2048, g_swz32[1][y & 31], 2048, VRAM_WORDS - 1};
    case CT16: return {bp * 128 + (y >> 6) * bw * 4096, g_swz16[0][y & 63], 4096, VRAM_WORDS * 2 - 1};
    case CT16S: return {bp * 128 + (y >> 6) * bw * 4096, g_swz16[1][y & 63], 4096, VRAM_WORDS * 2 - 1};
    case Z16: return {bp * 128 + (y >> 6) * bw * 4096, g_swz16[2][y & 63], 4096, VRAM_WORDS * 2 - 1};
    default: return {bp * 128 + (y >> 6) * bw * 4096, g_swz16[3][y & 63], 4096, VRAM_WORDS * 2 - 1};
    }
}
static __forceinline uint32_t fb_at(const FbRow &r, uint32_t x) { return (r.base + (x >> 6) * r.pgsz + r.sw[x & 63]) & r.mask; }

static bool fast_ok(const DrawCtx &d) {
    if (g_probe_x >= 0) return false;
    if (d.fpsm != CT32 && d.fpsm != CT24 && d.fpsm != CT16 && d.fpsm != CT16S) return false;
    return true;
}

template <bool DIRECT>
static __forceinline uint32_t fetch_t(const TexLevel &t, int u, int v) {   // u, v already wrapped
    if constexpr (DIRECT) {
        if (t.psm == CT32)   // inline the common case (frame buffer read back as a texture)
            return g_vram[(t.tbp * 64 + (((uint32_t)u >> 6) + ((uint32_t)v >> 5) * std::max(t.tbw, 1u)) * 2048 + g_swz32[0][v & 31][u & 63]) & (VRAM_WORDS - 1)];
        return direct_texel(t, u, v);
    }
    else return t.pix[(size_t)v * t.tw + u];
}
template <bool DIRECT>
static __forceinline uint32_t texel_t(const TexLevel &t, int u, int v) {
    return fetch_t<DIRECT>(t, wrap_coord(u, t.wms, t.tw, t.minu, t.maxu), wrap_coord(v, t.wmt, t.th, t.minv, t.maxv));
}
template <bool BIL, bool DIRECT>
static __forceinline Rgba sample_t(const TexLevel &t, float u, float v) {
    if constexpr (!BIL) return from32(texel_t<DIRECT>(t, (int)std::floor(u), (int)std::floor(v)));
    else {
        u -= 0.5f; v -= 0.5f;
        float fu0 = std::floor(u), fv0 = std::floor(v);
        int u0 = (int)fu0, v0 = (int)fv0;
        int fu = (int)((u - fu0) * 256), fv = (int)((v - fv0) * 256);
        int ua = wrap_coord(u0, t.wms, t.tw, t.minu, t.maxu), ub = wrap_coord(u0 + 1, t.wms, t.tw, t.minu, t.maxu);
        int va = wrap_coord(v0, t.wmt, t.th, t.minv, t.maxv), vb = wrap_coord(v0 + 1, t.wmt, t.th, t.minv, t.maxv);
        uint32_t a = fetch_t<DIRECT>(t, ua, va), b = fetch_t<DIRECT>(t, ub, va), c = fetch_t<DIRECT>(t, ua, vb), dd = fetch_t<DIRECT>(t, ub, vb);
        // per channel: top = p*(256-fu) + q*fu, bot likewise, (top*(256-fv) + bot*fv + 32768) >> 16
        const __m128i wu0 = _mm_set1_epi32(256 - fu), wu1 = _mm_set1_epi32(fu);
        const __m128i A = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)a)), B = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)b));
        const __m128i C = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)c)), D = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)dd));
        __m128i top = _mm_add_epi32(_mm_mullo_epi32(A, wu0), _mm_mullo_epi32(B, wu1));
        __m128i bot = _mm_add_epi32(_mm_mullo_epi32(C, wu0), _mm_mullo_epi32(D, wu1));
        __m128i res = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(top, _mm_set1_epi32(256 - fv)), _mm_mullo_epi32(bot, _mm_set1_epi32(fv))), _mm_set1_epi32(32768));
        res = _mm_srai_epi32(res, 16);
        Rgba o; _mm_storeu_si128((__m128i *)&o, res);
        return o;
    }
}
// sample() with the level descriptors built once per draw (same arithmetic)
struct MipSet { TexLevel lv[7]; float sc[7]; };
static void mip_setup(const TexState &t, MipSet &m) {
    m.lv[0] = TexLevel{t.pix, t.tw, t.th, t.wms, t.wmt, t.minu, t.maxu, t.minv, t.maxv, t.psm, t.tbp, t.tbw, t.texa};
    m.sc[0] = 1.0f;
    for (int lev = 1; lev < t.nlev && lev < 7; lev++) {
        TexLevel l{t.lpix[lev], std::max(t.tw >> lev, 1u), std::max(t.th >> lev, 1u), t.wms, t.wmt,
                   t.minu >> lev, t.maxu >> lev, t.minv >> lev, t.maxv >> lev, t.psm, t.mtbp[lev], t.mtbw[lev], t.texa};
        if (t.wms == 3) { l.minu = t.minu; l.maxu = t.maxu >> lev; }
        if (t.wmt == 3) { l.minv = t.minv; l.maxv = t.maxv >> lev; }
        m.lv[lev] = l; m.sc[lev] = 1.0f / (float)(1 << lev);
    }
}
static __forceinline Rgba sample_mip(const TexState &t, const MipSet &m, float u, float v, float q) {
    int lev = 0;
    bool bilinear = t.bilinear;
    float lod = t.lcm ? t.k : -fast_log2(std::fabs(q) + 1e-30f) * (float)(1 << t.lshift) + t.k;
    if (lod > 0) {
        bilinear = t.min_linear;
        lev = std::min((int)(lod + 0.5f), t.nlev - 1);
    }
    const TexLevel &l = m.lv[lev];
    if (lev) { u *= m.sc[lev]; v *= m.sc[lev]; }
    const bool direct = !l.pix;
    if (bilinear) return direct ? sample_t<true, true>(l, u, v) : sample_t<true, false>(l, u, v);
    return direct ? sample_t<false, true>(l, u, v) : sample_t<false, false>(l, u, v);
}
static thread_local MipSet t_mip;
template <bool TME, bool FGE, bool BIL, bool DIRECT, bool MIP = false>
static __forceinline Rgba shade_t(const DrawCtx &d, const TexLevel &tl, Rgba vc, float u, float v, int fog, float q = 1.0f) {
    Rgba c = vc;
    if constexpr (TME) {
        Rgba t;
        if constexpr (MIP) t = sample_mip(d.tex, t_mip, u, v, q);
        else t = sample_t<BIL, DIRECT>(tl, u, v);
        switch (d.tex.tfx) {
        case 0:
            c.r = (t.r * vc.r) >> 7; c.g = (t.g * vc.g) >> 7; c.b = (t.b * vc.b) >> 7;
            c.a = d.tex.tcc ? (t.a * vc.a) >> 7 : vc.a;
            break;
        case 1: c = t; if (!d.tex.tcc) c.a = vc.a; break;
        case 2:
            c.r = ((t.r * vc.r) >> 7) + vc.a; c.g = ((t.g * vc.g) >> 7) + vc.a; c.b = ((t.b * vc.b) >> 7) + vc.a;
            c.a = d.tex.tcc ? t.a + vc.a : vc.a;
            break;
        default:
            c.r = ((t.r * vc.r) >> 7) + vc.a; c.g = ((t.g * vc.g) >> 7) + vc.a; c.b = ((t.b * vc.b) >> 7) + vc.a;
            c.a = d.tex.tcc ? t.a : vc.a;
            break;
        }
        c.r = std::min(c.r, 255); c.g = std::min(c.g, 255); c.b = std::min(c.b, 255); c.a = std::min(c.a, 255);
    }
    if constexpr (FGE) {
        c.r = (fog * c.r + (255 - fog) * d.fogr) >> 8;
        c.g = (fog * c.g + (255 - fog) * d.fogg) >> 8;
        c.b = (fog * c.b + (255 - fog) * d.fogb) >> 8;
    }
    return c;
}

// Z test alone, ahead of shading: a pixel failing it writes nothing whatever the other tests say,
// so skipping its texture fetch is exact.
static __forceinline bool z_pass(const DrawCtx &d, uint32_t za, uint32_t z) {
    uint32_t zm = zmax(d.zpsm);
    if (z > zm) z = zm;
    bool z16 = d.zpsm == Z16 || d.zpsm == Z16S;
    uint32_t zd = (z16 ? vram16()[za] : g_vram[za]) & zm;
    return !(d.ztst == 0 || (d.ztst == 2 && z < zd) || (d.ztst == 3 && z <= zd));
}
// put_pixel with the address already known (fa: frame word/halfword, za: Z address)
template <bool ZTE, bool ABE, bool IS16>
static __forceinline void put_t(const DrawCtx &d, uint32_t fa, uint32_t za, uint32_t z, Rgba c) {
    bool fb_write = true, z_write = !d.zmsk, a_write = true;
    if (d.ate) {
        bool pass;
        int a = c.a, ref = (int)d.aref;
        switch (d.atst) {
        case 0: pass = false; break;
        case 1: pass = true; break;
        case 2: pass = a < ref; break;
        case 3: pass = a <= ref; break;
        case 4: pass = a == ref; break;
        case 5: pass = a >= ref; break;
        case 6: pass = a > ref; break;
        default: pass = a != ref; break;
        }
        if (!pass) {
            switch (d.afail) {
            case 0: return;
            case 1: z_write = false; break;
            case 2: fb_write = false; break;
            case 3: z_write = false; a_write = false; break;
            }
        }
    }
    uint32_t dst = IS16 ? vram16()[fa] : g_vram[fa];
    if (d.date) {
        uint32_t abit = IS16 ? (dst >> 15) & 1 : (dst >> 31) & 1;
        if (abit != (uint32_t)d.datm) return;
    }
    bool z16 = d.zpsm == Z16 || d.zpsm == Z16S;
    if constexpr (ZTE) {
        uint32_t zm = zmax(d.zpsm);
        if (z > zm) z = zm;
        uint32_t zd = (z16 ? vram16()[za] : g_vram[za]) & zm;
        bool zfail = d.ztst == 0 || (d.ztst == 2 && z < zd) || (d.ztst == 3 && z <= zd);
        if (zfail) return;
    } else z_write = false;
    if (fb_write) {
        int dr, dg, db, da;
        if constexpr (IS16) { dr = (dst & 31) << 3; dg = ((dst >> 5) & 31) << 3; db = ((dst >> 10) & 31) << 3; da = (dst & 0x8000) ? 0x80 : 0; }
        else { dr = dst & 255; dg = (dst >> 8) & 255; db = (dst >> 16) & 255; da = dst >> 24; }
        if (ABE && (!d.pabe || (c.a & 0x80))) {
            int ca = d.ac == 0 ? c.a : d.ac == 1 ? da : (int)d.afix;
            int sr[3] = {c.r, dr, 0}, sg[3] = {c.g, dg, 0}, sb[3] = {c.b, db, 0};
            int vr = (((sr[d.aa] - sr[d.ab]) * ca) >> 7) + sr[d.ad];
            int vg = (((sg[d.aa] - sg[d.ab]) * ca) >> 7) + sg[d.ad];
            int vb = (((sb[d.aa] - sb[d.ab]) * ca) >> 7) + sb[d.ad];
            if (d.colclamp) { c.r = std::clamp(vr, 0, 255); c.g = std::clamp(vg, 0, 255); c.b = std::clamp(vb, 0, 255); }
            else { c.r = vr & 255; c.g = vg & 255; c.b = vb & 255; }
        } else if (d.colclamp) {
            c.r = std::clamp(c.r, 0, 255); c.g = std::clamp(c.g, 0, 255); c.b = std::clamp(c.b, 0, 255);
        } else { c.r &= 255; c.g &= 255; c.b &= 255; }
        int a = std::clamp(c.a, 0, 255);
        if (d.fba) a |= 0x80;
        uint32_t mask = d.fbmsk;
        if constexpr (IS16) {
            uint32_t out = (uint32_t)(c.r >> 3) | ((uint32_t)(c.g >> 3) << 5) | ((uint32_t)(c.b >> 3) << 10) | ((a & 0x80) ? 0x8000u : 0);
            uint32_t m16 = ((mask >> 3) & 0x1F) | ((mask >> 6) & 0x3E0) | ((mask >> 9) & 0x7C00) | ((mask >> 16) & 0x8000);
            if (!a_write) m16 |= 0x8000;
            vram16()[fa] = (uint16_t)((out & ~m16) | (dst & m16));
        } else {
            uint32_t out = (uint32_t)c.r | ((uint32_t)c.g << 8) | ((uint32_t)c.b << 16) | ((uint32_t)a << 24);
            if (!a_write || d.fpsm == CT24) mask |= 0xFF000000u;
            g_vram[fa] = (out & ~mask) | (dst & mask);
        }
    }
    if (z_write) {
        if (z16) vram16()[za] = (uint16_t)z;
        else if (d.zpsm == Z24) g_vram[za] = (g_vram[za] & 0xFF000000u) | (z & 0xFFFFFF);
        else g_vram[za] = z;
    }
}

static inline TexLevel level0(const TexState &t) {
    return TexLevel{t.pix, t.tw, t.th, t.wms, t.wmt, t.minu, t.maxu, t.minv, t.maxv, t.psm, t.tbp, t.tbw, t.texa};
}

template <bool TME, bool FGE, bool ZTE, bool ABE, bool BIL, bool DIRECT, bool IS16, bool MIP = false>
static void tri_fast(const DrawCtx &d, const Vertex &v0, const Vertex &v1, const Vertex &v2) {
    float area = (v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y);
    if (area == 0 || !std::isfinite(area)) return;
    int miny = std::max((int)std::ceil(std::min({v0.y, v1.y, v2.y})), d.scy0);
    int maxy = std::min((int)std::ceil(std::max({v0.y, v1.y, v2.y})) - 1, d.scy1);
    if (miny > maxy) return;
    if (my_first_row(miny) > maxy) return;
    float inv = 1.0f / area;
    auto plane = [&](float a0, float a1, float a2) {
        Plane p;
        p.dx = ((a1 - a0) * (v2.y - v0.y) - (a2 - a0) * (v1.y - v0.y)) * inv;
        p.dy = ((a2 - a0) * (v1.x - v0.x) - (a1 - a0) * (v2.x - v0.x)) * inv;
        p.c = a0 - p.dx * v0.x - p.dy * v0.y;
        return p;
    };
    const bool gour = d.iip;
    Plane pr{}, pg{}, pb{}, pa{}, ps{}, pt{}, pq{}, pf{};
    if (gour) { pr = plane((float)v0.r, (float)v1.r, (float)v2.r); pg = plane((float)v0.g, (float)v1.g, (float)v2.g);
                pb = plane((float)v0.b, (float)v1.b, (float)v2.b); pa = plane((float)v0.a, (float)v1.a, (float)v2.a); }
    double zdx, zdy, zc;
    {
        double inv_d = 1.0 / (double)area;
        double a0 = v0.z, a1 = v1.z, a2 = v2.z;
        zdx = ((a1 - a0) * (v2.y - v0.y) - (a2 - a0) * (v1.y - v0.y)) * inv_d;
        zdy = ((a2 - a0) * (v1.x - v0.x) - (a1 - a0) * (v2.x - v0.x)) * inv_d;
        zc = a0 - zdx * v0.x - zdy * v0.y;
    }
    const bool fst = d.fst;
    if constexpr (TME) {
        if (fst) { ps = plane(v0.u, v1.u, v2.u); pt = plane(v0.v, v1.v, v2.v); }
        else { ps = plane(v0.s, v1.s, v2.s); pt = plane(v0.t, v1.t, v2.t); pq = plane(v0.q, v1.q, v2.q); }
    }
    if constexpr (FGE) pf = plane((float)v0.fog, (float)v1.fog, (float)v2.fog);
    const TexLevel tl = level0(d.tex);
    if constexpr (MIP) mip_setup(d.tex, t_mip);
    const float ftw = (float)d.tex.tw, fth = (float)d.tex.th;
    Rgba flat{v2.r, v2.g, v2.b, v2.a};
    const Vertex *vs[3] = {&v0, &v1, &v2};
    for (int y = my_first_row(miny); y <= maxy; y = ((y & g_band_mask) == g_band_mask) ? my_first_row(y + 1) : y + 1) {
        float fy = (float)y;
        float xl = -1e30f, xr = 1e30f;
        bool empty = false;
        for (int e = 0; e < 3; e++) {
            const Vertex &a = *vs[(e + 1) % 3], &b = *vs[(e + 2) % 3];
            float A = -(b.y - a.y), C = (b.x - a.x) * (fy - a.y);
            if (area < 0) { A = -A; C = -C; }
            if (A == 0) { if (C < 0) empty = true; continue; }
            float xb = a.x - C / A;
            if (A > 0) xl = std::max(xl, xb); else xr = std::min(xr, xb);
        }
        if (empty) continue;
        int x0 = std::max((int)std::ceil(xl), d.scx0), x1 = std::min((int)std::ceil(xr) - 1, d.scx1);
        if (x0 > x1) continue;
        t_px += (uint64_t)(x1 - x0 + 1);
        const FbRow fr = fb_row(d.fpsm, d.fbp, d.fbw, (uint32_t)y);
        const FbRow zr = ZTE ? fb_row(d.zpsm, d.zbp, d.fbw, (uint32_t)y) : fr;
        float fx = (float)x0;
        float r = pr.at(fx, fy), g = pg.at(fx, fy), bb = pb.at(fx, fy), al = pa.at(fx, fy);
        float s0 = ps.at(fx, fy), t0 = pt.at(fx, fy), q0 = pq.at(fx, fy), f0 = pf.at(fx, fy);
        double z = zdx * x0 + zdy * fy + zc;
        for (int x = x0; x <= x1; x++) {
            Rgba vc = gour ? Rgba{std::clamp((int)r, 0, 255), std::clamp((int)g, 0, 255), std::clamp((int)bb, 0, 255), std::clamp((int)al, 0, 255)} : flat;
            float u = 0, v = 0;
            if constexpr (TME) {
                if (fst) { u = s0; v = t0; }
                else { float q = q0 != 0 ? q0 : 1e-6f; u = s0 / q * ftw; v = t0 / q * fth; }
            }
            int fog = FGE ? std::clamp((int)f0, 0, 255) : 255;
            uint32_t zi = z <= 0 ? 0 : z >= 4294967295.0 ? 0xFFFFFFFFu : (uint32_t)z;
            const uint32_t za = ZTE ? fb_at(zr, (uint32_t)x) : 0;
            if (!ZTE || z_pass(d, za, zi))
                put_t<ZTE, ABE, IS16>(d, fb_at(fr, (uint32_t)x), za, zi,
                                      shade_t<TME, FGE, BIL, DIRECT, MIP>(d, tl, vc, u, v, fog, fst ? v2.q : q0));
            if (gour) { r += pr.dx; g += pg.dx; bb += pb.dx; al += pa.dx; }
            if constexpr (TME) { s0 += ps.dx; t0 += pt.dx; q0 += pq.dx; }
            if constexpr (FGE) f0 += pf.dx;
            z += zdx;
        }
    }
}

// draw_sprite's per-pixel part (the flat-fill case stays in draw_sprite)
template <bool TME, bool FGE, bool ZTE, bool ABE, bool BIL, bool DIRECT, bool IS16, bool MIP = false>
static void sprite_fast(const DrawCtx &d, const Vertex &a, const Vertex &b) {
    const Vertex &l = a.x <= b.x ? a : b, &r = a.x <= b.x ? b : a;
    const Vertex &tp = a.y <= b.y ? a : b, &bt = a.y <= b.y ? b : a;
    int ix0 = std::max((int)std::ceil(l.x), d.scx0), ix1 = std::min((int)std::ceil(r.x), d.scx1 + 1);
    int iy0 = std::max((int)std::ceil(tp.y), d.scy0), iy1 = std::min((int)std::ceil(bt.y), d.scy1 + 1);
    if (ix0 >= ix1 || iy0 >= iy1) return;
    float dx = r.x - l.x, dy = bt.y - tp.y;
    Rgba vc{b.r, b.g, b.b, b.a};
    float u0 = 0, du = 0, v0 = 0, dv = 0;
    if (d.fst) {
        du = dx > 0 ? (r.u - l.u) / dx : 0; u0 = l.u + (ix0 - l.x) * du;
        dv = dy > 0 ? (bt.v - tp.v) / dy : 0; v0 = tp.v + (iy0 - tp.y) * dv;
    } else {
        float q = b.q != 0 ? b.q : 1e-6f;
        float su = (float)d.tex.tw / q, sv = (float)d.tex.th / q;
        du = dx > 0 ? (r.s - l.s) / dx * su : 0; u0 = l.s * su + (ix0 - l.x) * du;
        dv = dy > 0 ? (bt.t - tp.t) / dy * sv : 0; v0 = tp.t * sv + (iy0 - tp.y) * dv;
    }
    const TexLevel tl = level0(d.tex);
    if constexpr (MIP) mip_setup(d.tex, t_mip);
    constexpr bool simple = !TME && !FGE;
    Rgba flat = vc;
    if constexpr (simple) flat = shade_t<TME, FGE, BIL, DIRECT>(d, tl, vc, 0, 0, b.fog);
    float v = v0;
    for (int y = iy0; y < iy1; y++, v += dv) {
        if (!my_row(y)) continue;
        t_px += (uint64_t)(ix1 - ix0);
        const FbRow fr = fb_row(d.fpsm, d.fbp, d.fbw, (uint32_t)y);
        const FbRow zr = ZTE ? fb_row(d.zpsm, d.zbp, d.fbw, (uint32_t)y) : fr;
        float u = u0;
        for (int x = ix0; x < ix1; x++, u += du) {
            const uint32_t za = ZTE ? fb_at(zr, (uint32_t)x) : 0;
            if (ZTE && !simple && !z_pass(d, za, b.z)) continue;
            put_t<ZTE, ABE, IS16>(d, fb_at(fr, (uint32_t)x), za, b.z,
                                  simple ? flat : shade_t<TME, FGE, BIL, DIRECT, MIP>(d, tl, vc, u, v, b.fog, b.q));
        }
    }
}

// dispatch tables: key = TME | FGE<<1 | ZTE<<2 | ABE<<3 | BIL<<4 | DIRECT<<5 | IS16<<6
typedef void (*TriFn)(const DrawCtx &, const Vertex &, const Vertex &, const Vertex &);
typedef void (*SprFn)(const DrawCtx &, const Vertex &, const Vertex &);
template <int K> static void tri_k(const DrawCtx &d, const Vertex &a, const Vertex &b, const Vertex &c) {
    tri_fast<(K & 1) != 0, (K & 2) != 0, (K & 4) != 0, (K & 8) != 0, (K & 16) != 0, (K & 32) != 0, (K & 64) != 0>(d, a, b, c);
}
template <int K> static void spr_k(const DrawCtx &d, const Vertex &a, const Vertex &b) {
    sprite_fast<(K & 1) != 0, (K & 2) != 0, (K & 4) != 0, (K & 8) != 0, (K & 16) != 0, (K & 32) != 0, (K & 64) != 0>(d, a, b);
}
template <int... K> static constexpr std::array<TriFn, sizeof...(K)> tri_table(std::integer_sequence<int, K...>) { return {&tri_k<K>...}; }
template <int... K> static constexpr std::array<SprFn, sizeof...(K)> spr_table(std::integer_sequence<int, K...>) { return {&spr_k<K>...}; }
static const std::array<TriFn, 128> g_tri_fast = tri_table(std::make_integer_sequence<int, 128>{});
static const std::array<SprFn, 128> g_spr_fast = spr_table(std::make_integer_sequence<int, 128>{});
// mipmapped: generic sample() per pixel, key = FGE | ZTE<<1 | ABE<<2 | IS16<<3
template <int K> static void trim_k(const DrawCtx &d, const Vertex &a, const Vertex &b, const Vertex &c) {
    tri_fast<true, (K & 1) != 0, (K & 2) != 0, (K & 4) != 0, false, false, (K & 8) != 0, true>(d, a, b, c);
}
template <int K> static void sprm_k(const DrawCtx &d, const Vertex &a, const Vertex &b) {
    sprite_fast<true, (K & 1) != 0, (K & 2) != 0, (K & 4) != 0, false, false, (K & 8) != 0, true>(d, a, b);
}
template <int... K> static constexpr std::array<TriFn, sizeof...(K)> trim_table(std::integer_sequence<int, K...>) { return {&trim_k<K>...}; }
template <int... K> static constexpr std::array<SprFn, sizeof...(K)> sprm_table(std::integer_sequence<int, K...>) { return {&sprm_k<K>...}; }
static const std::array<TriFn, 16> g_tri_mip = trim_table(std::make_integer_sequence<int, 16>{});
static const std::array<SprFn, 16> g_spr_mip = sprm_table(std::make_integer_sequence<int, 16>{});
static inline int fast_key(const DrawCtx &d) {
    bool tme = d.tme;
    return (tme ? 1 : 0) | (d.fge ? 2 : 0) | (d.zte ? 4 : 0) | (d.abe ? 8 : 0) | ((tme && d.tex.bilinear) ? 16 : 0) |
           ((tme && !d.tex.pix) ? 32 : 0) | ((d.fpsm == CT16 || d.fpsm == CT16S) ? 64 : 0);
}

// NPPL_RASTCHECK=<n>: every n-th fast draw is also rendered by the generic path (on this thread's
// rows of its bounding box) and the results compared.
static int g_rastcheck = -1;
static bool g_fast_raster = true;
static std::atomic<uint64_t> g_rastcheck_n, g_rastcheck_bad;
static inline bool sprite_flat(const DrawCtx &d) {
    return !d.tme && !d.abe && (!d.ate || d.atst == 1) && !d.date && (!d.zte || d.ztst <= 1) && g_probe_x < 0;
}
static void draw_triangle(const DrawCtx &d, const Vertex &v0, const Vertex &v1, const Vertex &v2);
static void draw_sprite(const DrawCtx &d, const Vertex &a, const Vertex &b);
static void run_fast(const DrawCmd &c, bool tri) {
    int k = fast_key(c.d);
    if (c.d.tme && c.d.tex.nlev > 1) {
        int m = ((k >> 1) & 7) | ((k >> 3) & 8);
        if (tri) g_tri_mip[m](c.d, c.v[0], c.v[1], c.v[2]); else g_spr_mip[m](c.d, c.v[0], c.v[1]);
        return;
    }
    if (tri) g_tri_fast[k](c.d, c.v[0], c.v[1], c.v[2]); else g_spr_fast[k](c.d, c.v[0], c.v[1]);
}
static void rast_check(const DrawCmd &c, bool tri) {
    const DrawCtx &d = c.d;
    int nv = tri ? 3 : 2;
    float mnx = c.v[0].x, mxx = mnx, mny = c.v[0].y, mxy = mny;
    for (int i = 1; i < nv; i++) { mnx = std::min(mnx, c.v[i].x); mxx = std::max(mxx, c.v[i].x); mny = std::min(mny, c.v[i].y); mxy = std::max(mxy, c.v[i].y); }
    int x0 = std::max((int)std::floor(mnx) - 1, d.scx0), x1 = std::min((int)std::ceil(mxx) + 1, d.scx1);
    int y0 = std::max((int)std::floor(mny) - 1, d.scy0), y1 = std::min((int)std::ceil(mxy) + 1, d.scy1);
    if (x0 > x1 || y0 > y1 || !std::isfinite(mnx + mxx + mny + mxy)) { run_fast(c, tri); return; }
    bool f16 = d.fpsm == CT16 || d.fpsm == CT16S, z16 = d.zpsm == Z16 || d.zpsm == Z16S;
    struct Px { uint32_t fa, za, f, z; };
    std::vector<Px> px;
    for (int y = y0; y <= y1; y++) {
        if (!my_row(y)) continue;
        for (int x = x0; x <= x1; x++) {
            Px p{fb_addr(d.fpsm, d.fbp, d.fbw, x, y), fb_addr(d.zpsm, d.zbp, d.fbw, x, y), 0, 0};
            p.f = f16 ? vram16()[p.fa] : g_vram[p.fa]; p.z = z16 ? vram16()[p.za] : g_vram[p.za];
            px.push_back(p);
        }
    }
    auto rd = [&](const Px &p, uint32_t &f, uint32_t &z) { f = f16 ? vram16()[p.fa] : g_vram[p.fa]; z = z16 ? vram16()[p.za] : g_vram[p.za]; };
    auto restore = [&]() { for (auto it = px.rbegin(); it != px.rend(); ++it) {
        if (z16) vram16()[it->za] = (uint16_t)it->z; else g_vram[it->za] = it->z;
        if (f16) vram16()[it->fa] = (uint16_t)it->f; else g_vram[it->fa] = it->f; } };
    if (tri) draw_triangle(d, c.v[0], c.v[1], c.v[2]); else draw_sprite(d, c.v[0], c.v[1]);
    std::vector<uint32_t> ref(px.size() * 2);
    for (size_t i = 0; i < px.size(); i++) rd(px[i], ref[i * 2], ref[i * 2 + 1]);
    restore();
    run_fast(c, tri);
    size_t bad = 0, first = ~(size_t)0;
    for (size_t i = 0; i < px.size(); i++) {
        uint32_t f, z; rd(px[i], f, z);
        if (f != ref[i * 2] || z != ref[i * 2 + 1]) { if (!bad) first = i; bad++; }
    }
    g_rastcheck_n++;
    if (bad) {
        uint64_t nb = ++g_rastcheck_bad;
        if (nb <= 40) {
            uint32_t f, z; rd(px[first], f, z);
            rt_log(LOG_ERR, "rastcheck: %s key=%02x fpsm=%x zpsm=%x tfx=%u fst=%d iip=%d ate=%d/%u date=%d: %zu/%zu px differ, first fb %08x vs %08x z %08x vs %08x",
                   tri ? "tri" : "spr", fast_key(d), d.fpsm, d.zpsm, d.tex.tfx, d.fst, d.iip, d.ate, d.atst, d.date, bad, px.size(),
                   f, ref[first * 2], z, ref[first * 2 + 1]);
        }
    }
}
static void draw_fast(const DrawCmd &c, bool tri) {
    if (g_rastcheck > 0) {
        static std::atomic<uint64_t> cnt;
        if (cnt++ % (uint64_t)g_rastcheck == 0) { rast_check(c, tri); return; }
    }
    run_fast(c, tri);
}

static void run_cmd(const DrawCmd &c) {
    const DrawCtx &d = c.d;
    if (d.prim == PRIM_BARRIER) {
        if (t_row_n > 1) { auto t0 = std::chrono::steady_clock::now(); g_barrier.wait(); g_barrier_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(); g_barrier_n++; }
        return;
    }
    switch (d.prim) {
    case 0:
        if (my_row((int)c.v[0].y))
            put_pixel(d, (int)c.v[0].x, (int)c.v[0].y, c.v[0].z,
                      shade(d, Rgba{c.v[0].r, c.v[0].g, c.v[0].b, c.v[0].a}, c.v[0].u, c.v[0].v, c.v[0].fog, c.v[0].q), c.v[0].fog);
        break;
    case 1: case 2: draw_line(d, c.v[0], c.v[1]); break;
    case 3: case 4: case 5:
        if (g_fast_raster && fast_ok(d)) draw_fast(c, true); else draw_triangle(d, c.v[0], c.v[1], c.v[2]);
        break;
    case 6:
        if (g_fast_raster && fast_ok(d) && !sprite_flat(d)) draw_fast(c, false); else draw_sprite(d, c.v[0], c.v[1]);
        break;
    }
}

// debug: pixels per (frame buffer, texture) pair, reported with the texture stats
struct PxStat { uint32_t key; uint64_t px, cmds; };
static PxStat g_pxstat[8][64];
static void pxstat_add(int k, const DrawCtx &d, uint64_t px) {
    uint32_t key = (d.fbp << 16) | (d.tme ? d.tex.tbp : 0xFFFF);
    PxStat *t = g_pxstat[k & 7];
    uint32_t h = (key * 2654435761u) >> 26;
    for (int i = 0; i < 64; i++) {
        PxStat &e = t[(h + i) & 63];
        if (e.key == key || !e.cmds) { e.key = key; e.px += px; e.cmds++; return; }
    }
}
static std::atomic<uint64_t> g_busy_ns;   // debug: total worker time spent rasterizing

// In-flight batches: a ring the workers consume in order. Ordering between draws in different
// batches comes from the in-stream barriers; the game thread only waits when the ring is full
// or when it needs VRAM that queued draws touch.
static const int NBATCH = 8;
struct Batch {
    std::vector<DrawCmd> cmds;
    std::vector<uint16_t> pg, rd;                 // pages counted in g_pg_cnt / g_rd_cnt
    std::vector<std::vector<uint32_t>> grave;
    std::atomic<int> pending{0};
};
static Batch g_ring[NBATCH];
static uint64_t g_submitted, g_reaped;          // game thread
static std::atomic<uint64_t> g_published{0};     // batches visible to the workers

static std::atomic<uint64_t> g_busy_k[8];
static void pxstat_report() {
    { char b[200]; int o = 0; for (int i = 0; i < 8 && i < g_nworkers; i++) o += snprintf(b + o, sizeof b - o, " %llu", (unsigned long long)(g_busy_k[i].exchange(0) / 1000000));
      rt_log(LOG_INFO, "raster busy per thread ms:%s | barriers %llu, %llu ms; px %llu", b, (unsigned long long)g_barrier_n.exchange(0) / std::max(g_nworkers, 1), (unsigned long long)g_barrier_ns.exchange(0) / 1000000, (unsigned long long)g_px_total.load()); }
    if (g_rastcheck > 0) rt_log(LOG_INFO, "rastcheck: %llu draws compared, %llu differ", (unsigned long long)g_rastcheck_n.load(), (unsigned long long)g_rastcheck_bad.load());
    rt_log(LOG_INFO, "raster: busy %llu ms over %d threads", (unsigned long long)(g_busy_ns.exchange(0) / 1000000), g_nworkers);
    gs_flush();
    std::vector<PxStat> all;
    for (auto &th : g_pxstat) for (auto &e : th) if (e.cmds) {
        bool f = false;
        for (auto &x : all) if (x.key == e.key) { x.px += e.px; x.cmds += e.cmds; f = true; break; }
        if (!f) all.push_back(e);
    }
    std::sort(all.begin(), all.end(), [](const PxStat &x, const PxStat &y) { return x.px > y.px; });
    std::string line;
    for (size_t i = 0; i < all.size() && i < 10; i++) {
        char b[64]; snprintf(b, sizeof b, " fb%04X/t%04X %lluk(%llu)", all[i].key >> 16, all[i].key & 0xFFFF, all[i].px / 1000, all[i].cmds);
        line += b;
    }
    rt_log(LOG_INFO, "px by target:%s", line.c_str());
    memset(g_pxstat, 0, sizeof g_pxstat);
}

void profiler_add_thread();
static void worker(int k, int n) {
    t_row_k = k; t_row_n = n;
    profiler_add_thread();
    uint64_t next = 0;
    for (;;) {
        if (g_published.load(std::memory_order_acquire) <= next) {
            std::unique_lock<std::mutex> lk(g_pool_mx);
            g_pool_cv.wait(lk, [&] { return g_published.load(std::memory_order_acquire) > next; });
        }
        Batch &b = g_ring[next % NBATCH];
        auto tb = std::chrono::steady_clock::now();
        for (const DrawCmd &c : b.cmds) { uint64_t p0 = t_px; run_cmd(c); pxstat_add(k, c.d, t_px - p0); }
        { uint64_t bn = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - tb).count(); g_busy_ns += bn; g_busy_k[k & 7] += bn; }
        g_px_total += t_px; t_px = 0;
        next++;
        if (b.pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lk(g_pool_mx);
            g_done_cv.notify_all();
        }
    }
}

// Retire finished batches (oldest first): release their page marks and dead textures.
static void reap() {
    while (g_reaped < g_submitted) {
        Batch &b = g_ring[g_reaped % NBATCH];
        if (b.pending.load(std::memory_order_acquire) != 0) break;
        for (uint16_t p : b.pg) g_pg_cnt[p]--;
        for (uint16_t p : b.rd) g_rd_cnt[p]--;
        b.pg.clear(); b.rd.clear(); b.grave.clear(); b.cmds.clear();
        g_reaped++;
    }
    g_prof_pixels = g_px_total;
}

// Wait until at most `keep` batches are still in flight.
static void wait_batches(uint64_t keep) {
    reap();
    if (g_submitted - g_reaped <= keep) return;
    auto t0 = std::chrono::steady_clock::now();
    {
        std::unique_lock<std::mutex> lk(g_pool_mx);
        g_done_cv.wait(lk, [&] {
            uint64_t done = g_reaped;
            while (done < g_submitted && g_ring[done % NBATCH].pending.load(std::memory_order_acquire) == 0) done++;
            return g_submitted - done <= keep;
        });
    }
    reap();
    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    g_prof_draw_ns += ns;
    g_wait_ns[g_wait_reason & 7] += ns; g_wait_n[g_wait_reason & 7]++;
    g_wait_reason = 0;
}

// Hand the queued draws to the rasterizer threads without waiting for them.
static void gs_submit() {
    if (g_cmds.empty()) return;
    if (g_nworkers <= 1) {
        auto t0 = std::chrono::steady_clock::now();
        for (const DrawCmd &c : g_cmds) run_cmd(c);
        g_cmds.clear();
        g_px_total += t_px; t_px = 0;
        g_prof_pixels = g_px_total;
        memset(g_pg_cur, 0, sizeof g_pg_cur);
        if (g_rd_any_cur) { memset(g_rd_cur, 0, sizeof g_rd_cur); g_rd_any_cur = false; }
        if (g_wr_since_any) { memset(g_wr_since, 0, sizeof g_wr_since); g_wr_since_any = false; }
        if (g_rd_since_any) { memset(g_rd_since, 0, sizeof g_rd_since); g_rd_since_any = false; }
        g_grave_cur.clear();
        g_prof_draw_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        return;
    }
    int reason = g_wait_reason;
    g_wait_reason = 3;
    wait_batches(NBATCH - 1);   // need a free slot
    g_wait_reason = reason;
    Batch &b = g_ring[g_submitted % NBATCH];
    b.cmds.swap(g_cmds); g_cmds.clear();
    for (uint32_t p = 0; p < NPAGES; p++) if (g_pg_cur[p]) { b.pg.push_back((uint16_t)p); g_pg_cnt[p]++; }
    memset(g_pg_cur, 0, sizeof g_pg_cur);
    if (g_rd_any_cur) {
        for (uint32_t p = 0; p < NPAGES; p++) if (g_rd_cur[p]) { b.rd.push_back((uint16_t)p); g_rd_cnt[p]++; }
        memset(g_rd_cur, 0, sizeof g_rd_cur); g_rd_any_cur = false;
    }
    b.grave.swap(g_grave_cur); g_grave_cur.clear();
    b.pending.store(g_nworkers, std::memory_order_release);
    g_submitted++;
    {
        std::lock_guard<std::mutex> lk(g_pool_mx);
        g_published.store(g_submitted, std::memory_order_release);
    }
    g_pool_cv.notify_all();
}

// Rasterize everything queued and wait. Needed before anything reads or writes VRAM the queue touches.
static void gs_flush() {
    gs_submit();
    if (g_nworkers > 1) {
        wait_batches(0);
        // everything has run: in-stream ordering starts afresh
        if (g_wr_since_any) { memset(g_wr_since, 0, sizeof g_wr_since); g_wr_since_any = false; }
        if (g_rd_since_any) { memset(g_rd_since, 0, sizeof g_rd_since); g_rd_since_any = false; }
    }
}

// Conservatively mark the frame/z pages a primitive can touch (texture-cache invalidation).
static void mark_prim_pages(const DrawCtx &d, const Vertex *v, int nv) {
    float x0 = v[0].x, x1 = v[0].x, y0 = v[0].y, y1 = v[0].y;
    for (int i = 1; i < nv; i++) { x0 = std::min(x0, v[i].x); x1 = std::max(x1, v[i].x); y0 = std::min(y0, v[i].y); y1 = std::max(y1, v[i].y); }
    int ix0, ix1, iy0, iy1;
    if (d.prim >= 3) {   // triangles / sprites cover pixels ceil(min) .. ceil(max) - 1 (draw_triangle, draw_sprite)
        ix0 = std::max((int)std::ceil(x0), d.scx0); ix1 = std::min((int)std::ceil(x1) - 1, d.scx1);
        iy0 = std::max((int)std::ceil(y0), d.scy0); iy1 = std::min((int)std::ceil(y1) - 1, d.scy1);
    } else {
        ix0 = std::max((int)std::floor(x0), d.scx0); ix1 = std::min((int)std::ceil(x1), d.scx1);
        iy0 = std::max((int)std::floor(y0), d.scy0); iy1 = std::min((int)std::ceil(y1), d.scy1);
    }
    if (ix0 > ix1 || iy0 > iy1) return;
    bool zw = d.zte && !d.zmsk;
    if (g_rd_since_any) {
        bool hit = false;
        for (int y = iy0 & ~31; y <= iy1 && !hit; y += 32)
            for (int x = ix0 & ~63; x <= ix1 && !hit; x += 64) {
                uint32_t fa = fb_addr(d.fpsm, d.fbp, d.fbw, (uint32_t)x, (uint32_t)y);
                fa = (d.fpsm == CT16 || d.fpsm == CT16S) ? fa >> 1 : fa;
                hit = g_rd_since[(fa >> 11) & (NPAGES - 1)] != 0;
            }
        if (hit) push_barrier();
    }
    // one sample per page cell (pages are 64 pixels wide and 32 or 64 rows tall)
    for (int y = iy0 & ~31; y <= iy1; y += 32)
        for (int x = ix0 & ~63; x <= ix1; x += 64) {
            uint32_t fa = fb_addr(d.fpsm, d.fbp, d.fbw, (uint32_t)x, (uint32_t)y);
            fa = (d.fpsm == CT16 || d.fpsm == CT16S) ? fa >> 1 : fa;
            mark_word(fa);
            g_pg_cur[(fa >> 11) & (NPAGES - 1)] = 1;
            g_page_drawn[(fa >> 11) & (NPAGES - 1)] = 1;
            g_wr_since[(fa >> 11) & (NPAGES - 1)] = 1; g_wr_since_any = true;
            {
                static const bool lg = getenv("NPPL_RECTLOG") != nullptr;
                if (lg && ((fa >> 11) & (NPAGES - 1)) == 350 && rt_once(0xD20000000ull ^ ((uint64_t)d.fbp << 20) ^ ((uint64_t)iy1 << 8) ^ (uint64_t)ix1 ^ ((uint64_t)d.prim << 40)))
                    rt_log(LOG_INFO, "draw page 350: fbp %04X fbw %u psm %02X prim %u box %d,%d-%d,%d tme %d tbp %04X", d.fbp, d.fbw, d.fpsm, d.prim, ix0, iy0, ix1, iy1, d.tme, d.tme ? d.tex.tbp : 0);
            }
            if (d.zte) {
                uint32_t za = fb_addr(d.zpsm, d.zbp, d.fbw, (uint32_t)x, (uint32_t)y);
                za = (d.zpsm == Z16 || d.zpsm == Z16S) ? za >> 1 : za;
                if (zw) mark_word(za);
                g_pg_cur[(za >> 11) & (NPAGES - 1)] = 1;
            }
        }
}

#include "gs_gpu.inc"

static void vertex_kick(uint64_t xyz, bool has_fog, bool draw) {
    DrawCtx d;
    build_ctx(d);
    uint64_t ofs = g_reg[R_XYOFFSET_1 + d.ctxt];
    Vertex v;
    v.x = ((float)(uint32_t)bits(xyz, 0, 16) - (float)(uint32_t)bits(ofs, 0, 16)) / 16.0f;
    v.y = ((float)(uint32_t)bits(xyz, 16, 16) - (float)(uint32_t)bits(ofs, 32, 16)) / 16.0f;
    v.z = has_fog ? (uint32_t)bits(xyz, 32, 24) : (uint32_t)(xyz >> 32);
    v.fog = has_fog ? (int)bits(xyz, 56, 8) : (int)bits(g_reg[R_FOG], 56, 8);
    uint64_t c = g_reg[R_RGBAQ];
    v.r = (int)bits(c, 0, 8); v.g = (int)bits(c, 8, 8); v.b = (int)bits(c, 16, 8); v.a = (int)bits(c, 24, 8);
    uint32_t qb = (uint32_t)(c >> 32), sb = (uint32_t)g_reg[R_ST], tb = (uint32_t)(g_reg[R_ST] >> 32);
    memcpy(&v.q, &qb, 4); memcpy(&v.s, &sb, 4); memcpy(&v.t, &tb, 4);
    v.u = (float)bits(g_reg[R_UV], 0, 14) / 16.0f; v.v = (float)bits(g_reg[R_UV], 16, 14) / 16.0f;

    int need;
    switch (d.prim) {
    case 0: need = 1; break;
    case 1: case 2: case 6: need = 2; break;
    case 3: case 4: case 5: need = 3; break;
    default: need = 99; break;
    }
    if (g_vcount < 3) g_vq[g_vcount++] = v;
    if (g_vcount < need) return;
    if (!draw && need == 3 && g_probe_x >= 0 && std::fabs((g_vq[1].x - g_vq[0].x) * (g_vq[2].y - g_vq[0].y) - (g_vq[2].x - g_vq[0].x) * (g_vq[1].y - g_vq[0].y)) > 0.5f) {
        const Vertex &a = g_vq[0], &b = g_vq[1], &cc = g_vq[2];
        float px = (float)g_probe_x, py = (float)g_probe_y;
        auto ed = [&](const Vertex &p0, const Vertex &p1) { return (p1.x - p0.x) * (py - p0.y) - (p1.y - p0.y) * (px - p0.x); };
        float e0 = ed(a, b), e1 = ed(b, cc), e2 = ed(cc, a);
        if ((e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0)) { t_cur_v = g_vq; probe_rec(d, 0, 0, Rgba{}, 0, 0, 4); t_cur_v = nullptr; }
    }
    if (draw && g_gpu) {
        g_draw_count++; g_prof_prims[d.prim]++;
        gpu::draw(d, g_vq, need > 3 ? 3 : need);
    } else if (draw) {
        g_draw_count++; g_prof_prims[d.prim]++;
        if (d.tme) {
            for (int tries = 0; tries < 2; tries++) {   // rebind everything if the cache got cleared midway
                uint32_t epoch = g_tex_epoch;
                for (int i = 1; i < d.tex.nlev; i++) {
                    TexState lv = d.tex;
                    lv.tbp = lv.mtbp[i]; lv.tbw = lv.mtbw[i];
                    lv.tw = std::max(d.tex.tw >> i, 1u); lv.th = std::max(d.tex.th >> i, 1u);
                    tex_bind(lv);
                    d.tex.lpix[i] = lv.pix;
                }
                tex_bind(d.tex);
                if (d.tex.direct) d.tex.nlev = 1;
                if (epoch == g_tex_epoch) break;
            }
        }
        DrawCmd cmd;
        cmd.d = d;
        int nv = need > 3 ? 3 : need;
        for (int i = 0; i < nv; i++) cmd.v[i] = g_vq[i];
        mark_prim_pages(d, cmd.v, nv);
        g_cmds.push_back(cmd);
        if (g_cmds.size() >= 4096) { g_wait_reason = 3; gs_submit(); }
    }
    switch (d.prim) {
    case 2: g_vq[0] = g_vq[1]; g_vcount = 1; break;           // line strip
    case 4: g_vq[0] = g_vq[1]; g_vq[1] = g_vq[2]; g_vcount = 2; break;   // triangle strip
    case 5: g_vq[1] = g_vq[2]; g_vcount = 2; break;           // triangle fan
    default: g_vcount = 0; break;
    }
}

// ================================================================ transfers
struct Xfer {
    bool active = false; uint32_t x = 0, y = 0; uint64_t acc = 0; int acc_bits = 0;
    uint32_t dbp = 0, dbw = 0, dpsm = 0, dx = 0, dy = 0, w = 0, h = 0;
};
static Xfer g_xfer;

static void xfer_start() {
    uint32_t dir = (uint32_t)bits(g_reg[R_TRXDIR], 0, 2);
    uint64_t bb = g_reg[R_BITBLTBUF], pos = g_reg[R_TRXPOS], reg = g_reg[R_TRXREG];
    uint32_t w = (uint32_t)bits(reg, 0, 12), h = (uint32_t)bits(reg, 32, 12);
    if (dir == 0) {
        sync_rect((uint32_t)bits(bb, 56, 6), (uint32_t)bits(bb, 32, 14), (uint32_t)bits(bb, 48, 6),
                  (uint32_t)bits(pos, 32, 11), (uint32_t)bits(pos, 48, 11), w, h, 1);
        for_rect_pages((uint32_t)bits(bb, 56, 6), (uint32_t)bits(bb, 32, 14), (uint32_t)bits(bb, 48, 6),
                       (uint32_t)bits(pos, 32, 11), (uint32_t)bits(pos, 48, 11), w, h, [](uint32_t p) { if (g_page_drawn[p]) { g_page_drawn[p] = 0; g_undraw_gen++; } return false; });
        if (g_gpu) gpu_sync_rect((uint32_t)bits(bb, 56, 6), (uint32_t)bits(bb, 32, 14), (uint32_t)bits(bb, 48, 6),
                                 (uint32_t)bits(pos, 32, 11), (uint32_t)bits(pos, 48, 11), w, h, true);
        g_xfer = Xfer{}; g_xfer.active = true;
        g_xfer.dbp = (uint32_t)bits(bb, 32, 14); g_xfer.dbw = (uint32_t)bits(bb, 48, 6); g_xfer.dpsm = (uint32_t)bits(bb, 56, 6);
        g_xfer.dx = (uint32_t)bits(pos, 32, 11); g_xfer.dy = (uint32_t)bits(pos, 48, 11); g_xfer.w = w; g_xfer.h = h;
        // texture-cache invalidation for the whole destination (the fast paths below skip per-pixel marks)
        for_rect_pages(g_xfer.dpsm, g_xfer.dbp, g_xfer.dbw, g_xfer.dx, g_xfer.dy, w, h, [](uint32_t p) { g_page_gen[p] = g_write_gen; return false; });
        if (!w || !h) g_xfer.active = false;
        return;
    }
    g_wait_reason = 4;
    gs_flush();
    if (dir == 2) {
        uint32_t sbp = (uint32_t)bits(bb, 0, 14), sbw = (uint32_t)bits(bb, 16, 6), spsm = (uint32_t)bits(bb, 24, 6);
        uint32_t dbp = (uint32_t)bits(bb, 32, 14), dbw = (uint32_t)bits(bb, 48, 6), dpsm = (uint32_t)bits(bb, 56, 6);
        uint32_t sx = (uint32_t)bits(pos, 0, 11), sy = (uint32_t)bits(pos, 16, 11), dx = (uint32_t)bits(pos, 32, 11), dy = (uint32_t)bits(pos, 48, 11);
        if (g_gpu) { gpu_sync_rect(spsm, sbp, sbw, sx, sy, w, h, false); gpu_sync_rect(dpsm, dbp, dbw, dx, dy, w, h, true); }
        std::vector<uint32_t> tmp((size_t)w * h);
        for (uint32_t y = 0; y < h; y++)
            for (uint32_t x = 0; x < w; x++) tmp[(size_t)y * w + x] = read_raw(spsm, sbp, sbw, (sx + x) & 2047, (sy + y) & 2047);
        for (uint32_t y = 0; y < h; y++)
            for (uint32_t x = 0; x < w; x++) write_raw(dpsm, dbp, dbw, (dx + x) & 2047, (dy + y) & 2047, tmp[(size_t)y * w + x]);
    } else if (dir == 1 && rt_once(0x700000001ull)) rt_log(LOG_WARN, "GS: local->host transfer not supported");
}

static void xfer_data(uint64_t data) {
    if (!g_xfer.active) return;
    Xfer &X = g_xfer;
    uint32_t dpsm = X.dpsm;
    if (X.acc_bits == 0 && (dpsm == CT32 || dpsm == CT16 || dpsm == CT16S || dpsm == T8 || dpsm == T4)) {
        int pb = dpsm == CT32 ? 32 : dpsm == T8 ? 8 : dpsm == T4 ? 4 : 16;
        uint32_t m = pb == 32 ? 0xFFFFFFFFu : (1u << pb) - 1;
        for (int i = 0; i < 64 / pb; i++) {
            uint32_t v = (uint32_t)(data >> (i * pb)) & m;
            uint32_t x = (X.dx + X.x) & 2047, y = (X.dy + X.y) & 2047;
            switch (dpsm) {
            case CT32: g_vram[addr32(X.dbp, X.dbw, x, y, blockTable32)] = v; break;
            case CT16: vram16()[addr16(X.dbp, X.dbw, x, y, blockTable16)] = (uint16_t)v; break;
            case CT16S: vram16()[addr16(X.dbp, X.dbw, x, y, blockTable16S)] = (uint16_t)v; break;
            case T8: vram8()[addr8(X.dbp, X.dbw, x, y)] = (uint8_t)v; break;
            default: { uint32_t n = addr4(X.dbp, X.dbw, x, y); uint8_t &b = vram8()[n >> 1]; b = (n & 1) ? (uint8_t)((b & 0x0F) | (v << 4)) : (uint8_t)((b & 0xF0) | v); break; }
            }
            if (++X.x >= X.w) { X.x = 0; if (++X.y >= X.h) { X.active = false; return; } }
        }
        return;
    }
    uint64_t bb = g_reg[R_BITBLTBUF], pos = g_reg[R_TRXPOS], reg = g_reg[R_TRXREG];
    uint32_t dbp = (uint32_t)bits(bb, 32, 14), dbw = (uint32_t)bits(bb, 48, 6);
    uint32_t dx = (uint32_t)bits(pos, 32, 11), dy = (uint32_t)bits(pos, 48, 11);
    uint32_t w = (uint32_t)bits(reg, 0, 12), h = (uint32_t)bits(reg, 32, 12);
    int pb = psm_bits(dpsm);
    // feed 64 bits; consume pixels of pb bits (24-bit pixels may straddle words)
    int avail = 64;
    uint64_t src = data;
    while (avail > 0) {
        int take = std::min(avail, pb - g_xfer.acc_bits);
        uint64_t part = take == 64 ? src : (src & ((1ull << take) - 1));
        g_xfer.acc |= part << g_xfer.acc_bits;
        g_xfer.acc_bits += take;
        src = take == 64 ? 0 : src >> take;
        avail -= take;
        if (g_xfer.acc_bits == pb) {
            if (g_xfer.y < h) write_raw(dpsm, dbp, dbw, (dx + g_xfer.x) & 2047, (dy + g_xfer.y) & 2047, (uint32_t)g_xfer.acc);
            g_xfer.acc = 0; g_xfer.acc_bits = 0;
            if (++g_xfer.x >= w) { g_xfer.x = 0; g_xfer.y++; }
            if (g_xfer.y >= h) { g_xfer.active = false; return; }
        }
    }
}

// ================================================================ register writes
static void gs_write(uint32_t r, uint64_t v) {
    r &= 0xFF;
    switch (r) {
    case R_PRIM: g_reg[r] = v; g_vcount = 0; return;
    case R_XYZF2: g_reg[r] = v; vertex_kick(v, true, true); return;
    case R_XYZ2: g_reg[r] = v; vertex_kick(v, false, true); return;
    case R_XYZF3: g_reg[r] = v; vertex_kick(v, true, false); return;
    case R_XYZ3: g_reg[r] = v; vertex_kick(v, false, false); return;
    case R_TEX0_1: case R_TEX0_2: g_reg[r] = v; clut_load(v); return;
    case R_TEX2_1: case R_TEX2_2: {   // TEX2 updates PSM / CLUT fields of TEX0
        uint64_t &t0 = g_reg[r - R_TEX2_1 + R_TEX0_1];
        uint64_t mask = (0x3Full << 20) | (0x7FFFFFFFull << 37);
        t0 = (t0 & ~mask) | (v & mask);
        clut_load(t0);
        g_reg[r] = v; return;
    }
    case R_TRXDIR: g_reg[r] = v; xfer_start(); return;
    case R_HWREG: xfer_data(v); return;
    case R_SIGNAL: g_csr_events |= 1; return;
    case R_FINISH: g_csr_events |= 2; return;
    case R_LABEL: return;
    default: g_reg[r] = v; return;
    }
}

// ================================================================ GIF
struct GifPath { uint32_t nloop = 0; uint32_t nreg = 0, reg_i = 0, flg = 0; uint64_t regs = 0; bool eop = false; bool active = false; };
static GifPath g_path[3];

static void gif_packed(GifPath &p, uint64_t lo, uint64_t hi) {
    uint32_t desc = (uint32_t)((p.regs >> (p.reg_i * 4)) & 15);
    switch (desc) {
    case 0x0: gs_write(R_PRIM, lo & 0x7FF); break;
    case 0x1: {
        uint64_t q = g_reg[R_RGBAQ] & 0xFFFFFFFF00000000ull;
        g_reg[R_RGBAQ] = (lo & 0xFF) | (((lo >> 32) & 0xFF) << 8) | ((hi & 0xFF) << 16) | (((hi >> 32) & 0xFF) << 24) | q;
        break;
    }
    case 0x2:
        g_reg[R_ST] = lo;
        g_reg[R_RGBAQ] = (g_reg[R_RGBAQ] & 0xFFFFFFFFull) | (hi << 32);
        break;
    case 0x3: g_reg[R_UV] = (lo & 0x3FFF) | (((lo >> 32) & 0x3FFF) << 16); break;
    case 0x4: {
        uint64_t v = (lo & 0xFFFF) | (((lo >> 32) & 0xFFFF) << 16) | (((hi >> 4) & 0xFFFFFF) << 32) | (((hi >> 36) & 0xFF) << 56);
        gs_write((hi >> 47) & 1 ? R_XYZF3 : R_XYZF2, v);
        break;
    }
    case 0x5: {
        uint64_t v = (lo & 0xFFFF) | (((lo >> 32) & 0xFFFF) << 16) | ((hi & 0xFFFFFFFF) << 32);
        gs_write((hi >> 47) & 1 ? R_XYZ3 : R_XYZ2, v);
        break;
    }
    case 0xA: g_reg[R_FOG] = ((hi >> 36) & 0xFF) << 56; break;
    case 0xE: gs_write((uint32_t)(hi & 0xFF), lo); break;
    case 0xF: break;
    default: gs_write(desc, lo); break;   // TEX0_1, TEX0_2, CLAMP_1, CLAMP_2, XYZF3, XYZ3...
    }
}

static void gif_feed_impl(int path, const uint8_t *data, uint32_t qwords) {
    GifPath &p = g_path[path];
    for (uint32_t q = 0; q < qwords; q++) {
        uint64_t lo, hi;
        memcpy(&lo, data + q * 16, 8);
        memcpy(&hi, data + q * 16 + 8, 8);
        if (!p.active || p.nloop == 0) {
            // new GIF tag
            p.nloop = (uint32_t)(lo & 0x7FFF);
            p.eop = (lo >> 15) & 1;
            p.flg = (uint32_t)((lo >> 58) & 3);
            p.nreg = (uint32_t)((lo >> 60) & 15); if (!p.nreg) p.nreg = 16;
            p.regs = hi;
            p.reg_i = 0;
            p.active = p.nloop != 0;
            if ((lo >> 46) & 1) gs_write(R_PRIM, (lo >> 47) & 0x7FF);
            continue;
        }
        if (p.flg == 0) {   // PACKED
            gif_packed(p, lo, hi);
            if (++p.reg_i >= p.nreg) { p.reg_i = 0; p.nloop--; }
        } else if (p.flg == 1) {   // REGLIST: two 64-bit register values per qword
            uint64_t w[2] = {lo, hi};
            for (int k = 0; k < 2 && p.nloop; k++) {
                uint32_t desc = (uint32_t)((p.regs >> (p.reg_i * 4)) & 15);
                if (desc != 0xE && desc != 0xF) gs_write(desc, w[k]);
                if (++p.reg_i >= p.nreg) { p.reg_i = 0; p.nloop--; }
            }
        } else {   // IMAGE
            xfer_data(lo); xfer_data(hi);
            p.nloop--;
        }
        if (p.nloop == 0) p.active = false;
    }
}

// VU1 XGKICK: walk GIF tags in VU memory until EOP.
static void xgkick_impl(const uint8_t *mem, uint32_t addr, uint32_t size) {
    GifPath &p = g_path[GIF_PATH1];
    p = GifPath{};
    for (uint32_t n = 0; n < size / 16; n++) {
        gif_feed_impl(GIF_PATH1, mem + addr, 1);
        addr = (addr + 16) & (size - 1);
        if (!p.active && p.eop) return;
    }
}

// ================================================================ privileged registers / display
uint64_t gs_priv_reg(uint32_t off) { return g_priv[(off & 0x1FFF) >> 3]; }

uint64_t g_gs_flips;   // DISPFB base changes (game frames shown)
static void priv_write_impl(uint32_t off, uint64_t v) {
    uint32_t i = (off & 0x1FFF) >> 3;
    if ((i == (0x70 >> 3) || i == (0x90 >> 3)) && ((g_priv[i] ^ v) & 0x1FF)) g_gs_flips++;
    g_priv[i] = v;
    if (rt_once(0x600000000ull | off)) rt_log(LOG_INFO, "GS priv reg %04X <- %016llX", off, (unsigned long long)v);
}


// ================================================================ GS thread
// Everything the EE sends to the GS (GIF paths 1-3, privileged register writes, CSR acks and
// vblanks) goes through a single-producer queue to a GS thread, which runs the GIF parser, the
// draw setup and the texture cache and feeds the rasterizer threads - so the EE, VU1 and the GS
// front end overlap like they do on the console. Reading CSR (FINISH/SIGNAL polls) waits for the
// queue to drain. NPPL_MTGS=0 runs the GS on the EE thread instead.
void display_present();   // display.cpp
void profiler_add_thread();
enum : uint8_t { GQ_SKIP, GQ_FEED, GQ_KICK1, GQ_PRIV, GQ_CSRACK, GQ_CSRSET, GQ_VSYNC };
struct GqHdr { uint8_t type, path; uint16_t pad; uint32_t n; uint64_t v; };   // 16 bytes, then n qwords
static_assert(sizeof(GqHdr) == 16, "queue header must be one qword");
static const uint64_t GQ_SIZE = 64ull << 20;
static uint8_t *g_gq;
static std::atomic<uint64_t> g_gq_wr{0}, g_gq_rd{0};   // monotonic byte counts
static std::atomic<int> g_gq_sleep{0};
static std::atomic<uint64_t> g_vsync_posted{0}, g_vsync_done{0};
static bool g_mtgs;
static uint64_t g_gq_wr_local;   // producer's private copy of the write position
static uint64_t g_drains, g_drain_ns;

static void gq_wait_space(uint64_t need) {
    for (int i = 0; g_gq_wr_local + need - g_gq_rd.load(std::memory_order_acquire) > GQ_SIZE; i++) {
        if (i < 2000) _mm_pause(); else SwitchToThread();
    }
}
static void gq_publish() {
    g_gq_wr.store(g_gq_wr_local, std::memory_order_seq_cst);
    if (g_gq_sleep.load(std::memory_order_seq_cst)) WakeByAddressSingle((void *)&g_gq_wr);
}
// Reserve room for a header + n qwords, contiguous in the buffer; returns the write pointer.
static uint8_t *gq_reserve(uint32_t n) {
    uint64_t need = 16 + (uint64_t)n * 16;
    uint64_t off = g_gq_wr_local % GQ_SIZE, room = GQ_SIZE - off;
    if (room < need) {   // pad to the end of the buffer
        gq_wait_space(room);
        GqHdr h{GQ_SKIP, 0, 0, 0, 0};
        memcpy(g_gq + off, &h, 16);
        g_gq_wr_local += room;
        off = 0;
    }
    gq_wait_space(need);
    return g_gq + off;
}
static void gq_put(uint8_t type, uint8_t path, uint64_t v, uint32_t n32, const uint8_t *a, uint32_t na, const uint8_t *b = nullptr, uint32_t nb = 0) {
    uint8_t *w = gq_reserve(na + nb);
    GqHdr h{type, path, 0, na + nb, v};
    if (!na && !nb) h.n = n32;
    memcpy(w, &h, 16);
    if (na) memcpy(w + 16, a, (size_t)na * 16);
    if (nb) memcpy(w + 16 + (size_t)na * 16, b, (size_t)nb * 16);
    g_gq_wr_local += 16 + (uint64_t)(na + nb) * 16;
    gq_publish();
}
// Wait until the GS thread has processed everything queued.
static void gs_sync() {
    if (!g_mtgs) return;
    if (g_gq_rd.load(std::memory_order_acquire) == g_gq_wr_local) return;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; g_gq_rd.load(std::memory_order_acquire) != g_gq_wr_local; i++) {
        if (i < 4000) _mm_pause(); else SwitchToThread();
    }
    g_drains++;
    g_drain_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}
static void gs_thread() {
    profiler_add_thread();
    uint64_t rd = 0;
    for (;;) {
        uint64_t wr = g_gq_wr.load(std::memory_order_acquire);
        if (wr == rd) {
            for (int i = 0; i < 20000 && (wr = g_gq_wr.load(std::memory_order_acquire)) == rd; i++) _mm_pause();
            if (wr == rd) {
                g_gq_sleep.store(1, std::memory_order_seq_cst);
                while ((wr = g_gq_wr.load(std::memory_order_seq_cst)) == rd) WaitOnAddress((volatile void *)&g_gq_wr, &rd, sizeof rd, INFINITE);
                g_gq_sleep.store(0, std::memory_order_relaxed);
            }
        }
        while (rd != wr) {
            const uint8_t *c = g_gq + rd % GQ_SIZE;
            GqHdr h; memcpy(&h, c, 16);
            uint64_t len = 16;
            switch (h.type) {
            case GQ_SKIP: len = GQ_SIZE - rd % GQ_SIZE; break;
            case GQ_FEED: gif_feed_impl(h.path, c + 16, h.n); len += (uint64_t)h.n * 16; break;
            case GQ_KICK1: g_path[GIF_PATH1] = GifPath{}; gif_feed_impl(GIF_PATH1, c + 16, h.n); len += (uint64_t)h.n * 16; break;
            case GQ_PRIV: priv_write_impl(h.n, h.v); break;
            case GQ_CSRACK: g_csr_events &= ~((uint32_t)h.v & 0x1F); break;
            case GQ_CSRSET: g_csr_events |= (uint32_t)h.v; break;
            case GQ_VSYNC: display_present(); g_vsync_done.fetch_add(1, std::memory_order_release); break;
            }
            rd += len;
            g_gq_rd.store(rd, std::memory_order_release);
        }
    }
}

void gif_feed(int path, const uint8_t *data, uint32_t qwords) {
    if (!g_mtgs) { gif_feed_impl(path, data, qwords); return; }
    while (qwords) {
        uint32_t n = std::min(qwords, 65536u);
        gq_put(GQ_FEED, (uint8_t)path, 0, 0, data, n);
        data += (size_t)n * 16; qwords -= n;
    }
}
void vu1_xgkick_hook(const uint8_t *mem, uint32_t addr, uint32_t size) {
    if (!g_mtgs) { xgkick_impl(mem, addr, size); return; }
    // find the packet length (tags until EOP), then queue it as one linear copy
    uint32_t total = 0, a = addr, lim = size / 16;
    for (;;) {
        uint64_t tag; memcpy(&tag, mem + (a & (size - 1)), 8);
        uint32_t nloop = tag & 0x7FFF, flg = (tag >> 58) & 3, nreg = (tag >> 60) & 15; if (!nreg) nreg = 16;
        uint32_t qw = flg == 0 ? nloop * nreg : flg == 1 ? (nloop * nreg + 1) / 2 : nloop;
        total += qw + 1; a = (a + (qw + 1) * 16) & (size - 1);
        if (((tag >> 15) & 1) || total >= lim) break;
    }
    total = std::min(total, lim);
    uint32_t first = std::min(total, (size - addr) / 16);
    gq_put(GQ_KICK1, 0, 0, 0, mem + addr, first, mem, total - first);
}
void gs_priv_write(uint32_t off, uint64_t v) {
    if (!g_mtgs) { priv_write_impl(off, v); return; }
    gq_put(GQ_PRIV, 0, v, off, nullptr, 0);
}
void gs_set_csr_bits(uint32_t b) { if (g_mtgs) gq_put(GQ_CSRSET, 0, b, 0, nullptr, 0); else g_csr_events |= b; }
uint32_t gs_csr_events() { gs_sync(); return g_csr_events; }
void gs_csr_ack(uint32_t v) { if (g_mtgs) gq_put(GQ_CSRACK, 0, v, 0, nullptr, 0); else g_csr_events &= ~(v & 0x1F); }
// vblank: present the displayed frame (on the GS thread, in order with the drawing). The EE may
// run at most two vblanks ahead of the GS.
void gs_vsync() {
    if (!g_mtgs) { display_present(); return; }
    uint64_t posted = g_vsync_posted.fetch_add(1) + 1;
    gq_put(GQ_VSYNC, 0, 0, 0, nullptr, 0);
    for (int i = 0; posted - g_vsync_done.load(std::memory_order_acquire) > 2; i++) {
        if (i < 4000) _mm_pause(); else SwitchToThread();
    }
}
bool gs_gpu_active() { return g_gpu; }
bool gs_gpu_present() { if (!g_gpu) return false; gpu::present(); return g_gpu; }
void gs_drain_stats(uint64_t &n, uint64_t &ns) { n = g_drains; ns = g_drain_ns; g_drains = 0; g_drain_ns = 0; }

bool gs_read_display(uint32_t *out, int max_w, int max_h, int &w, int &h) {
    uint64_t pmode = g_priv[0], dispfb = g_priv[0x90 >> 3], display = g_priv[0xA0 >> 3];
    if (!(pmode & 2) && (pmode & 1)) { dispfb = g_priv[0x70 >> 3]; display = g_priv[0x80 >> 3]; }
    uint32_t fbp = (uint32_t)bits(dispfb, 0, 9) * 32, fbw = (uint32_t)bits(dispfb, 9, 6), psm = (uint32_t)bits(dispfb, 15, 5);
    uint32_t dbx = (uint32_t)bits(dispfb, 32, 11), dby = (uint32_t)bits(dispfb, 43, 11);
    uint32_t magh = (uint32_t)bits(display, 23, 4) + 1, magv = (uint32_t)bits(display, 27, 2) + 1;
    uint32_t dw = (uint32_t)bits(display, 32, 12) + 1, dh = (uint32_t)bits(display, 44, 11) + 1;
    w = (int)(dw / magh); h = (int)(dh / magv);
    if (h > 512) h /= 2;
    if (w <= 0 || h <= 0 || !fbw) return false;
    w = std::min(w, max_w); h = std::min(h, max_h);
    {   // wait only if queued draws still write the displayed pages (usually the back buffer is being drawn)
        bool busy = false;
        reap();
        for_rect_pages(psm, fbp, fbw, dbx, dby, (uint32_t)w, (uint32_t)h, [&](uint32_t pg) { return busy = (g_pg_cur[pg] | g_pg_cnt[pg]) != 0; });
        // Busy: keep showing the previous picture rather than stall emulation (the finished
        // frame's last passes are usually still queued right after the flip). Wait if it persists.
        static int skipped;
        if (busy) {
            gs_submit();
            static const int max_skip = getenv("NPPL_DISPWAIT") ? 0 : 2;
            if (skipped < max_skip) { skipped++; return false; }
            g_wait_reason = 5; gs_flush();
        }
        skipped = 0;
    }
    if ((psm == CT32 || psm == CT24) && dbx + w <= 2048 && dby + h <= 2048) {
        for (int y = 0; y < h; y++) {
            const FbRow r = fb_row(psm, fbp, fbw, dby + y);
            uint32_t *o = out + (size_t)y * max_w;
            for (int x = 0; x < w; x++) {
                uint32_t v = g_vram[fb_at(r, dbx + x)];
                o[x] = ((v & 255) << 16) | (v & 0xFF00) | ((v >> 16) & 255);
            }
        }
        return true;
    }
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            uint32_t v = read_raw(psm, fbp, fbw, dbx + x, dby + y);
            uint32_t r, g, b;
            if (psm_bits(psm) == 16) { r = (v & 31) << 3; g = ((v >> 5) & 31) << 3; b = ((v >> 10) & 31) << 3; }
            else { r = v & 255; g = (v >> 8) & 255; b = (v >> 16) & 255; }
            out[y * max_w + x] = (r << 16) | (g << 8) | b;   // BGRA for GDI
        }
    return true;
}

static bool check_perm(const uint8_t *t, int n) {
    std::vector<int> seen(n, 0);
    for (int i = 0; i < n; i++) { if (t[i] >= n || seen[t[i]]++) return false; }
    return true;
}

void gs_init() {
    build_swizzle();
    {   // the tables must reproduce the block/column formula exactly
        bool ok = true;
        for (uint32_t y = 0; y < 128 && ok; y++)
            for (uint32_t x = 0; x < 256 && ok; x++) {
                uint32_t page = (x >> 6) + (y >> 5) * 3, page16 = (x >> 6) + (y >> 6) * 3;
                ok &= addr32(0x40, 3, x, y, blockTable32) == ((0x40 + page * 32 + blockTable32[(y >> 3) & 3][(x >> 3) & 7]) * 64 + columnTable32[y & 7][x & 7]);
                ok &= addr32(0x40, 3, x, y, blockTableZ32) == ((0x40 + page * 32 + blockTableZ32[(y >> 3) & 3][(x >> 3) & 7]) * 64 + columnTable32[y & 7][x & 7]);
                ok &= addr16(0x40, 3, x, y, blockTableZ16S) == ((0x40 + page16 * 32 + blockTableZ16S[(y >> 3) & 7][(x >> 4) & 3]) * 128 + columnTable16[y & 7][x & 15]);
                ok &= addr16(0x40, 3, x, y, blockTable16S) == ((0x40 + page16 * 32 + blockTable16S[(y >> 3) & 7][(x >> 4) & 3]) * 128 + columnTable16[y & 7][x & 15]);
            }
        if (!ok) rt_log(LOG_ERR, "GS: swizzle tables do not match the block/column formula");
    }
    if (const char *p = getenv("NPPL_PROBE")) sscanf(p, "%d,%d", &g_probe_x, &g_probe_y);
    if (const char *p = getenv("NPPL_RASTCHECK")) g_rastcheck = atoi(p);
    if (const char *p = getenv("NPPL_FASTRASTER")) g_fast_raster = atoi(p) != 0;
    g_vram = (uint32_t *)calloc(VRAM_WORDS, 4);
    g_nworkers = (int)std::min(8u, std::max(1u, std::thread::hardware_concurrency() - 1));
    if (const char *p = getenv("NPPL_RTHREADS")) g_nworkers = std::max(1, atoi(p));
    if (const char *p = getenv("NPPL_BAND")) { g_band_sh = std::clamp(atoi(p), 1, 6); g_band_mask = (1 << g_band_sh) - 1; }
    g_barrier.n = g_nworkers;
    if (g_nworkers > 1) for (int k = 0; k < g_nworkers; k++) std::thread(worker, k, g_nworkers).detach();
    rt_log(LOG_INFO, "GS: %d rasterizer threads", g_nworkers);
    bool ok = check_perm(&columnTable32[0][0], 64) && check_perm(&columnTable16[0][0], 128) &&
              check_perm(&blockTable32[0][0], 32) && check_perm(&blockTable16[0][0], 32) && check_perm(&blockTable8[0][0], 32);
    std::vector<int> seen8(256, 0), seen4(512, 0);
    for (int y = 0; y < 16; y++) for (int x = 0; x < 16; x++) if (seen8[columnTable8[y][x]]++) ok = false;
    for (int y = 0; y < 16; y++) for (int x = 0; x < 32; x++) if (columnTable4[y][x] >= 512 || seen4[columnTable4[y][x]]++) ok = false;
    if (!ok) rt_log(LOG_ERR, "GS: swizzle tables are not permutations");
    g_mtgs = !(getenv("NPPL_MTGS") && atoi(getenv("NPPL_MTGS")) == 0);
    g_gpu = !(getenv("NPPL_GPU") && atoi(getenv("NPPL_GPU")) == 0);   // -sw / NPPL_GPU=0: software renderer
    if (g_mtgs) {
        g_gq = (uint8_t *)VirtualAlloc(nullptr, GQ_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        std::thread(gs_thread).detach();
        rt_log(LOG_INFO, "GS: running on its own thread");
    }
}
