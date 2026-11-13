// AUDIO.IRX replacement (the game's own sound driver, sid 0x01234567).
//
// Protocol, from the EE SoundClient:
//   fn 1 opcodes: 0 free wave | 1-3 listener vectors (x1024) | 4 master volume (0-16383) | 5 ?
//         6 attach (voice, wave, loop) | 8 start | 7, 9 stop | 10-12, 15 voice params (unknown)
//         13 voice position (x1024) | 14 voice volume (0-16383) | 16 rate Hz (0 = paused)
//         17 doppler pitch (x1024) | 18 volume factor (x1024)
//   fn 1  command stream (async): words [op, args...]; reply = 8 bytes of status per voice
//         (byte 4 = voice state: 1 playing, 0 finished)
//   fn 2  allocate wave {index, alloc size, data size, rate, fmt}; reply {IOP address, bytes used}
//         -> the EE then SifSetDma's the sample data to that address
//   fn 3  wave uploaded {index, size}; for streamed waves (fmt 0x1xx) this is the next chunk:
//         the EE always DMAs to the start of the wave's block, so the data is copied out to a
//         FIFO here. Status bit 2 requests another chunk (the EE checks it every frame).
//   fn 5  reset            fn 6 version (0x101A)
//   fn 7  bytes used       fn 8 largest free block
// Sample data is SPU2 ADPCM (16-byte frames of 28 samples); a 2-channel wave stores each
// channel as one contiguous half. It is decoded to PCM here, when the upload completes.
// Wave memory is a 2 MB pool outside IOP RAM, reached through IOP addresses 0x400000-0x5FFFFF.
#include "rt_internal.h"
#include "sif.h"
#include "audio.h"
#include <algorithm>
#include <map>
#include <mutex>
#include <vector>

static const uint32_t POOL_BASE = 0x00400000, POOL_SIZE = 0x00200000;
static std::vector<uint8_t> g_pool(POOL_SIZE);
static std::map<uint32_t, uint32_t> g_blocks;   // offset -> size (allocated)

std::mutex g_audio_mx;
AudioWave g_waves[AUDIO_MAX_WAVES];
AudioVoice g_voices[AUDIO_MAX_VOICES];
AudioListener g_listener;
int g_audio_nvoices = 32;

uint8_t *iop_audio_ram(uint32_t iop_addr, uint32_t size) {
    iop_addr &= 0x0FFFFFFF;
    if (iop_addr < POOL_BASE || iop_addr + size > POOL_BASE + POOL_SIZE) return nullptr;
    return g_pool.data() + (iop_addr - POOL_BASE);
}
const uint8_t *audio_pool() { return g_pool.data(); }

static uint32_t pool_used() {
    uint32_t u = 0;
    for (auto &b : g_blocks) u += b.second;
    return u;
}
static uint32_t pool_largest_free() {
    uint32_t prev = 0, best = 0;
    for (auto &b : g_blocks) { best = std::max(best, b.first - prev); prev = b.first + b.second; }
    return std::max(best, POOL_SIZE - prev);
}
static int64_t pool_alloc(uint32_t size) {
    size = (size + 63) & ~63u;
    uint32_t prev = 0;
    for (auto &b : g_blocks) {
        if (b.first - prev >= size) { g_blocks[prev] = size; return prev; }
        prev = b.first + b.second;
    }
    if (POOL_SIZE - prev >= size) { g_blocks[prev] = size; return prev; }
    return -1;
}
static void pool_free(uint32_t off) { g_blocks.erase(off); }

static void free_wave(uint32_t idx) {
    if (idx >= AUDIO_MAX_WAVES || !g_waves[idx].valid) return;
    for (auto &v : g_voices) if (v.playing && v.wave == (int)idx) v.playing = false;
    pool_free(g_waves[idx].offset);
    g_waves[idx] = AudioWave{};
}

// PS ADPCM. Decodes `bytes` of frames from src, appending one sample per output stride.
// Stops after a frame with the end flag when stop_at_end is set; reports the loop-start sample.
static const int k_f0[5] = {0, 60, 115, 98, 122}, k_f1[5] = {0, 0, -52, -55, -60};
static size_t adpcm_decode(const uint8_t *src, size_t bytes, int16_t *out, size_t stride, int hist[2],
                           bool stop_at_end, uint32_t *loop_start) {
    size_t n = 0;
    for (size_t f = 0; f + 16 <= bytes; f += 16) {
        const uint8_t *fr = src + f;
        int shift = fr[0] & 15, filt = std::min(fr[0] >> 4, 4), flags = fr[1];
        if (flags == 7) { if (stop_at_end) break; continue; }   // end-of-data filler
        if ((flags & 4) && loop_start) *loop_start = (uint32_t)n;
        if (shift > 12) shift = 9;
        for (int k = 0; k < 28; k++) {
            int nib = (fr[2 + k / 2] >> ((k & 1) * 4)) & 15;
            int smp = (int16_t)(nib << 12) >> shift;
            smp += (hist[0] * k_f0[filt] + hist[1] * k_f1[filt] + 32) / 64;
            smp = std::clamp(smp, -32768, 32767);
            hist[1] = hist[0]; hist[0] = smp;
            if (out) out[n * stride] = (int16_t)smp;
            n++;
        }
        if ((flags & 1) && stop_at_end) break;
    }
    return n;
}

static void decode_wave(AudioWave &wv) {
    int ch = wv.channels();
    uint32_t plane = wv.bytes / ch;
    const uint8_t *src = g_pool.data() + wv.offset;
    size_t frames = 0;
    for (int c = 0; c < ch; c++) {   // decode lengths can differ per plane; use the longest
        int h[2] = {};
        frames = std::max(frames, adpcm_decode(src + c * plane, plane, nullptr, 1, h, true, nullptr));
    }
    wv.pcm.assign(frames * ch, 0);
    wv.loop_frame = 0;
    for (int c = 0; c < ch; c++) {
        int h[2] = {};
        adpcm_decode(src + c * plane, plane, wv.pcm.data() + c, ch, h, true, c == 0 ? &wv.loop_frame : nullptr);
    }
    if (wv.loop_frame >= frames) wv.loop_frame = 0;
}

static void stream_chunk(AudioWave &wv, uint32_t size) {
    int ch = wv.channels();
    uint32_t plane = std::min(size, wv.alloc) / ch;
    const uint8_t *src = g_pool.data() + wv.offset;
    size_t frames = plane / 16 * 28;
    std::vector<int16_t> tmp(frames * ch, 0);
    size_t got = 0;
    for (int c = 0; c < ch; c++)
        got = std::max(got, adpcm_decode(src + c * plane, plane, tmp.data() + c, ch, wv.hist[c], false, nullptr));
    wv.fifo.insert(wv.fifo.end(), tmp.begin(), tmp.begin() + got * ch);
}

static int32_t arg(const uint32_t *w, size_t n, size_t i) { return i < n ? (int32_t)w[i] : 0; }

// Number of argument words for each opcode (op 0-18).
static const int k_nargs[19] = {1, 3, 3, 3, 1, 1, 3, 1, 1, 1, 2, 2, 2, 4, 2, 2, 2, 2, 2};

static void run_commands(const uint32_t *w, size_t n) {
    size_t i = 0;
    while (i < n) {
        uint32_t op = w[i++];
        if (op > 18) {
            if (rt_once(0xA0000000ull | op)) rt_log(LOG_WARN, "audio: unknown opcode %u, dropping rest of batch", op);
            return;
        }
        const uint32_t *a = w + i;
        size_t na = n - i;
        i += k_nargs[op];
        if (g_log_level >= LOG_TRACE) {
            char b[96]; int o = snprintf(b, sizeof b, "audio: op %u", op);
            for (int k = 0; k < k_nargs[op]; k++) o += snprintf(b + o, sizeof b - o, " %d", arg(a, na, k));
            rt_log(LOG_TRACE, "%s", b);
        }
        int vi = arg(a, na, 0);
        AudioVoice *v = (vi >= 0 && vi < AUDIO_MAX_VOICES) ? &g_voices[vi] : nullptr;
        switch (op) {
        case 0: free_wave((uint32_t)arg(a, na, 0)); break;
        case 1: for (int k = 0; k < 3; k++) g_listener.pos[k] = arg(a, na, k) / 1024.0f; break;
        case 2: for (int k = 0; k < 3; k++) g_listener.fwd[k] = arg(a, na, k) / 1024.0f; break;
        case 3: for (int k = 0; k < 3; k++) g_listener.up[k] = arg(a, na, k) / 1024.0f; break;
        case 4: g_listener.param4 = arg(a, na, 0); break;
        case 5: g_listener.param5 = arg(a, na, 0); break;
        case 6:   // attach (voice, wave, loop); playback starts with op 8
            if (v) {
                int wi = arg(a, na, 1);
                if (wi < 0 || wi >= AUDIO_MAX_WAVES) break;   // the wave may be allocated after this
                *v = AudioVoice{};
                v->wave = wi; v->loop = arg(a, na, 2) != 0;
            }
            break;
        case 8: if (v && v->wave >= 0) { v->playing = true; v->pos = 0; } break;   // start
        case 7: case 9: if (v) v->playing = false; break;                          // stop
        case 10: if (v) v->p10 = arg(a, na, 1); break;
        case 11: if (v) v->p11 = arg(a, na, 1); break;
        case 12: if (v) v->p12 = arg(a, na, 1); break;
        case 13: if (v) { for (int k = 0; k < 3; k++) v->pos3d[k] = arg(a, na, 1 + k) / 1024.0f; v->is3d = true; } break;
        case 14: if (v) v->p14 = arg(a, na, 1); break;
        case 15: if (v) v->p15 = arg(a, na, 1); break;
        case 16: if (v) v->rate = arg(a, na, 1); break;
        case 17: if (v) v->pitch = arg(a, na, 1) / 1024.0f; break;
        case 18: if (v) v->volume = arg(a, na, 1) / 1024.0f; break;
        }
    }
}

static void srv_audio(SifRpcCall &c) {
    const uint8_t *in = g_iop_ram + (c.buff & 0x1FFFFF);
    uint32_t w[8] = {};
    memcpy(w, in, std::min<uint32_t>(c.send_size, sizeof w));
    std::lock_guard<std::mutex> lk(g_audio_mx);
    switch (c.fno) {
    case 1: {   // command batch; reply with voice states
        std::vector<uint32_t> cmd(c.send_size / 4);
        memcpy(cmd.data(), in, cmd.size() * 4);
        run_commands(cmd.data(), cmd.size());
        uint32_t nv = std::min<uint32_t>(c.recv_size / 8, AUDIO_MAX_VOICES);
        g_audio_nvoices = (int)nv;
        std::vector<uint8_t> st(c.recv_size, 0);
        for (uint32_t i = 0; i < nv; i++) {
            AudioVoice &v = g_voices[i];
            uint32_t p = (uint32_t)v.pos;   // samples played
            memcpy(&st[i * 8], &p, 4);
            uint8_t flags = v.playing ? 1 : 0;
            if (v.playing && v.wave >= 0 && g_waves[v.wave].stream) {
                AudioWave &wv = g_waves[v.wave];
                bool low = wv.fifo.size() < (size_t)wv.alloc / 16 * 28 * 2;   // keep about two chunks queued
                if (wv.req_age >= 0) wv.req_age++;
                if (low && (wv.req_age < 0 || wv.req_age > 30)) { flags |= 4; wv.req_age = 0; }
                if (wv.fifo.empty() && wv.req_age > 120) { v.playing = false; flags = 0; }   // the EE ran out of data
            }
            st[i * 8 + 4] = flags;
        }
        if (c.recv_addr) sif_write_ee(c.recv_addr, st.data(), c.recv_size);
        return;
    }
    case 2: {   // allocate wave
        uint32_t idx = w[0];
        if (idx < AUDIO_MAX_WAVES && g_waves[idx].valid) free_wave(idx);
        int64_t off = pool_alloc(w[1]);
        uint32_t reply[2] = {0, 0};
        if (off >= 0 && idx < AUDIO_MAX_WAVES) {
            AudioWave &wv = g_waves[idx];
            wv.valid = true; wv.uploaded = false;
            wv.offset = (uint32_t)off; wv.alloc = w[1]; wv.bytes = w[2]; wv.rate = (int)w[3];
            wv.fmt_lo = (uint16_t)w[4]; wv.fmt_hi = (uint16_t)(w[4] >> 16);
            wv.stream = (wv.fmt_lo & 0x100) != 0;
            reply[0] = POOL_BASE + (uint32_t)off;
        }
        reply[1] = pool_used();
        static int n; if (n++ < 8)
            rt_log(LOG_INFO, "audio: alloc wave %u size %u data %u rate %u fmt %04X/%04X -> %08X", idx, w[1], w[2], w[3],
                   w[4] & 0xFFFF, w[4] >> 16, reply[0]);
        if (c.recv_addr) sif_write_ee(c.recv_addr, reply, std::min<uint32_t>(c.recv_size, 8));
        return;
    }
    case 3: {   // upload finished
        if (w[0] < AUDIO_MAX_WAVES && g_waves[w[0]].valid) {
            AudioWave &wv = g_waves[w[0]];
            wv.uploaded = true;
            if (wv.stream) {
                stream_chunk(wv, w[1]);
                wv.req_age = -1;
                return;
            }
            decode_wave(wv);
            if (g_log_level >= LOG_TRACE)
                rt_log(LOG_TRACE, "audio: wave %u decoded: %zu frames x%d, loop at %u", w[0], wv.pcm.size() / wv.channels(), wv.channels(), wv.loop_frame);
        }
        return;
    }
    case 5:
        for (uint32_t i = 0; i < AUDIO_MAX_WAVES; i++) free_wave(i);
        for (auto &v : g_voices) v = AudioVoice{};
        return;
    case 6: { uint32_t v = 0x101A; if (c.recv_addr) sif_write_ee(c.recv_addr, &v, std::min<uint32_t>(c.recv_size, 4)); return; }
    case 7: { uint32_t v = pool_used(); if (c.recv_addr) sif_write_ee(c.recv_addr, &v, std::min<uint32_t>(c.recv_size, 4)); return; }
    case 8: { uint32_t v = pool_largest_free(); if (c.recv_addr) sif_write_ee(c.recv_addr, &v, std::min<uint32_t>(c.recv_size, 4)); return; }
    }
    if (rt_once(0xF20000000ull | c.fno)) rt_log(LOG_INFO, "audio: unhandled fn %u (send %u recv %u)", c.fno, c.send_size, c.recv_size);
    if (c.recv_addr && c.recv_size) { std::vector<uint8_t> z(c.recv_size, 0); sif_write_ee(c.recv_addr, z.data(), c.recv_size); }
}

void iop_audio_init() {
    for (auto &v : g_voices) v = AudioVoice{};
    sif_register_server(0x01234567, "audio", srv_audio);
    audio_out_init();
}
