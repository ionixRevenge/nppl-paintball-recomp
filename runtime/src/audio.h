// Host-side sound state shared by the AUDIO.IRX replacement (iop_audio.cpp) and the mixer (audio_out.cpp).
#pragma once
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

enum { AUDIO_MAX_WAVES = 1024, AUDIO_MAX_VOICES = 64 };

struct AudioWave {
    bool valid = false, uploaded = false;
    uint32_t offset = 0, alloc = 0, bytes = 0;   // in the wave pool
    int rate = 0;
    uint16_t fmt_lo = 0, fmt_hi = 0;             // as sent by the EE (see iop_audio.cpp)
    // Streamed waves (fmt_lo & 0x100): the pool block is only a transfer buffer; each fn 3
    // appends that many bytes to this FIFO, and status bit 2 asks the EE for the next chunk.
    bool stream = false;
    std::deque<int16_t> fifo;                    // decoded PCM, interleaved
    int hist[2][2] = {};                         // stream ADPCM decoder state per channel
    std::vector<int16_t> pcm;                    // non-stream waves: decoded PCM, interleaved
    uint32_t loop_frame = 0;                     // from the ADPCM loop-start flag
    int req_age = -1;                            // replies since the last data request (-1 = none)
    int channels() const { return (fmt_lo & 0xFF) == 2 ? 2 : 1; }
};

struct AudioVoice {
    bool playing = false, paused = false, loop = false, is3d = false;
    int wave = -1;
    double pos = 0;          // in sample frames (for streams: frames played)
    double frac = 0;         // streams: fraction between the two head frames
    int rate = 0;            // Hz (op 16)
    float pitch = 1.0f;      // doppler multiplier (op 17)
    float volume = 1.0f;     // op 18
    float pos3d[3] = {};
    int32_t p10 = 0, p11 = 0, p12 = 0, p14 = 16383, p15 = 0;
};

struct AudioListener {
    float pos[3] = {}, fwd[3] = {0, 0, 1}, up[3] = {0, 1, 0};
    int32_t param4 = 16383, param5 = 0;
};

extern std::mutex g_audio_mx;
extern AudioWave g_waves[AUDIO_MAX_WAVES];
extern AudioVoice g_voices[AUDIO_MAX_VOICES];
extern AudioListener g_listener;
extern int g_audio_nvoices;

const uint8_t *audio_pool();
uint8_t *iop_audio_ram(uint32_t iop_addr, uint32_t size);   // nullptr if not in the wave pool
void audio_out_init();
