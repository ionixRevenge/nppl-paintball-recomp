// Runtime-internal declarations (not used by generated code).
#pragma once
#include "ps2rt.h"
double rt_timescale();   // main.cpp
uint64_t rt_emu_ns();    // main.cpp: emulated time since boot
#include <cstdio>
#include <cstdarg>

extern uint8_t g_vu0_mem[0x1000], g_vu0_micro[0x1000];
extern uint8_t g_vu1_mem[0x4000], g_vu1_micro[0x4000];
extern uint8_t g_iop_ram[0x200000];

void vu_fbrst(uint32_t v);
void vu0_run_micro(R5900 *c, uint32_t addr);

// hardware
void hw_init();
uint32_t hw_read32(uint32_t a);
void hw_write32(uint32_t a, uint32_t v);
uint64_t hw_read64(uint32_t a);
void hw_write64(uint32_t a, uint64_t v);
void hw_vsync_tick();

// kernel
void kernel_init(R5900 *main_ctx);
[[noreturn]] void rt_exit(int code);

// logging
enum { LOG_ERR = 0, LOG_WARN = 1, LOG_INFO = 2, LOG_TRACE = 3 };
extern int g_log_level;
void rt_log(int level, const char *fmt, ...);
bool rt_once(uint64_t key);   // true the first time a key is seen
