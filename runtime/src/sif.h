// SIF RPC server interface for high-level IOP module implementations.
#pragma once
#include <cstdint>

struct SifRpcCall {
    uint32_t sid, fno;
    uint32_t buff;        // IOP-side buffer holding the request data (in g_iop_ram)
    uint32_t send_size;
    uint32_t recv_addr;   // EE address the reply data goes to
    uint32_t recv_size;
};
typedef void (*SifRpcHandler)(SifRpcCall &call);

void sif_init();
void sif_register_server(uint32_t sid, const char *name, SifRpcHandler fn);
bool sif_deliver_one();
bool sif_has_pending();
void sif_write_ee(uint32_t ee_addr, const void *data, uint32_t size);
