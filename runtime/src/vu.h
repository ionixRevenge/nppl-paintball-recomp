// VU0/VU1 micro-mode interpreter.
#pragma once
#include "ps2rt.h"

struct VuState {
    // VU0 micro mode shares the COP2 register file with the EE (macro mode), so
    // register storage lives in an R5900 context: vf[], vi[] (incl. control regs), vacc, vq.
    R5900 *regs;
    uint8_t *mem; uint32_t mem_mask;      // data memory (VU0 4 KB, VU1 16 KB)
    uint8_t *micro; uint32_t micro_mask;  // micro memory (VU0 4 KB, VU1 16 KB)
    int index;                            // 0 or 1
    float p;                              // EFU result register
    uint32_t tpc;                         // program counter (bytes)
    // pipelined Q / P (results become visible after their latency)
    float q_pending, p_pending;
    int64_t q_ready, p_ready, cycle;
    int64_t vf_ready[32][4];              // cycle at which each VF field's pending FMAC result is usable
    // VU1 double-buffer registers (set by VIF1)
    uint32_t top, itop;
};

extern VuState g_vu0, g_vu1;
void vu_init();
void vu_execute(VuState &vu, uint32_t start_pc);   // run until an E-bit program end
void vu_micro_dirty(int index);   // micro memory changed: recheck compiled code
void vu1_xgkick_hook(const uint8_t *packet_base, uint32_t addr, uint32_t mem_size);  // gs.cpp
