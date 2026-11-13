// Software mixer + Win32 waveOut output (48 kHz, 16-bit stereo).
#include "rt_internal.h"
#include "audio.h"
#define NOMINMAX
#include <windows.h>
#include <mmsystem.h>
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#pragma comment(lib, "winmm.lib")

static const int OUT_RATE = 48000, BLOCK = 512, NBUF = 6;
bool g_audio_mute;
const char *g_audio_wav;          // -wav FILE: also write the mixed output to a .wav (for testing)
static FILE *g_wav;
static uint32_t g_wav_bytes;

static void wav_header() {
    uint32_t hdr[11] = {0x46464952, 36 + g_wav_bytes, 0x45564157, 0x20746D66, 16, 0x00020001, (uint32_t)OUT_RATE,
                        (uint32_t)OUT_RATE * 4, 0x00100004, 0x61746164, g_wav_bytes};
    fseek(g_wav, 0, SEEK_SET); fwrite(hdr, 4, 11, g_wav); fseek(g_wav, 0, SEEK_END); fflush(g_wav);
}

// Decoded PCM (see iop_audio.cpp), interleaved by channel.
static inline void fetch(const AudioWave &w, uint32_t frame, float &l, float &r) {
    int ch = w.channels();
    uint32_t frames = (uint32_t)(w.pcm.size() / ch);
    if (frame >= frames) { l = r = 0; return; }
    const int16_t *s = w.pcm.data() + (size_t)frame * ch;
    l = s[0] / 32768.0f;
    r = ch == 2 ? s[1] / 32768.0f : l;
}

static void mix(int16_t *out, int n) {
    std::vector<float> acc(n * 2, 0.0f);
    {
        std::lock_guard<std::mutex> lk(g_audio_mx);
        for (int vi = 0; vi < AUDIO_MAX_VOICES; vi++) {
            AudioVoice &v = g_voices[vi];
            if (!v.playing || v.paused || v.wave < 0) continue;
            const AudioWave &w = g_waves[v.wave];
            if (!w.valid || !w.uploaded) continue;
            int ch = w.channels();
            uint32_t frames = (uint32_t)(w.pcm.size() / ch);
            if (!frames && !w.stream) { v.playing = false; continue; }
            double step = (double)(v.rate > 0 ? v.rate : w.rate) * v.pitch / OUT_RATE;
            if (step <= 0) continue;   // rate 0 = paused
            float gain = std::max(0.0f, v.volume) * (v.p14 / 16383.0f) * (g_listener.param4 / 16383.0f), gl = gain, gr = gain;
            if (v.is3d) {   // simple distance rolloff and pan relative to the listener
                float d[3] = {v.pos3d[0] - g_listener.pos[0], v.pos3d[1] - g_listener.pos[1], v.pos3d[2] - g_listener.pos[2]};
                float dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                float att = 1.0f / (1.0f + dist * 0.05f);
                const float *f = g_listener.fwd, *u = g_listener.up;
                float rgt[3] = {u[1] * f[2] - u[2] * f[1], u[2] * f[0] - u[0] * f[2], u[0] * f[1] - u[1] * f[0]};
                float pan = dist > 0.001f ? (d[0] * rgt[0] + d[1] * rgt[1] + d[2] * rgt[2]) / dist : 0.0f;
                pan = std::clamp(pan, -1.0f, 1.0f);
                gl = gain * att * std::sqrt(0.5f * (1.0f - pan));
                gr = gain * att * std::sqrt(0.5f * (1.0f + pan));
            }
            if (w.stream) {
                AudioWave &sw = g_waves[v.wave];
                for (int i = 0; i < n; i++) {
                    if (sw.fifo.size() < (size_t)ch) break;   // underrun: silence until the next chunk
                    float l0 = sw.fifo[0] / 32768.0f, r0 = sw.fifo[ch - 1] / 32768.0f, l1 = l0, r1 = r0;
                    if (sw.fifo.size() >= (size_t)ch * 2) { l1 = sw.fifo[ch] / 32768.0f; r1 = sw.fifo[2 * ch - 1] / 32768.0f; }
                    float fr = (float)v.frac;
                    acc[i * 2] += (l0 + (l1 - l0) * fr) * gl;
                    acc[i * 2 + 1] += (r0 + (r1 - r0) * fr) * gr;
                    v.frac += step;
                    while (v.frac >= 1.0 && sw.fifo.size() >= (size_t)ch) {
                        sw.fifo.erase(sw.fifo.begin(), sw.fifo.begin() + ch);
                        v.frac -= 1.0; v.pos += 1;
                    }
                }
                continue;
            }
            for (int i = 0; i < n; i++) {
                uint32_t f0 = (uint32_t)v.pos;
                if (f0 >= frames) {
                    if (v.loop) { v.pos -= frames - w.loop_frame; f0 = (uint32_t)v.pos; }
                    else { v.playing = false; break; }
                }
                float frac = (float)(v.pos - f0), l0, r0, l1, r1;
                fetch(w, f0, l0, r0);
                fetch(w, (f0 + 1 < frames) ? f0 + 1 : (v.loop ? w.loop_frame : f0), l1, r1);
                acc[i * 2] += (l0 + (l1 - l0) * frac) * gl;
                acc[i * 2 + 1] += (r0 + (r1 - r0) * frac) * gr;
                v.pos += step;
            }
        }
    }
    for (int i = 0; i < n * 2; i++) {
        float s = acc[i];
        s = s / (1.0f + std::fabs(s) * 0.25f);   // soft limiter
        out[i] = (int16_t)std::clamp(s * 32767.0f, -32768.0f, 32767.0f);
    }
    if (g_wav) {
        fwrite(out, 4, n, g_wav); g_wav_bytes += n * 4;
        if ((g_wav_bytes / (n * 4)) % 64 == 0) wav_header();
    }
    if (g_audio_mute) std::fill(out, out + n * 2, (int16_t)0);
}

static void out_thread() {
    WAVEFORMATEX fmt{};
    fmt.wFormatTag = WAVE_FORMAT_PCM; fmt.nChannels = 2; fmt.nSamplesPerSec = OUT_RATE;
    fmt.wBitsPerSample = 16; fmt.nBlockAlign = 4; fmt.nAvgBytesPerSec = OUT_RATE * 4;
    HANDLE ev = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    HWAVEOUT h;
    if (waveOutOpen(&h, WAVE_MAPPER, &fmt, (DWORD_PTR)ev, 0, CALLBACK_EVENT) != MMSYSERR_NOERROR) {
        rt_log(LOG_WARN, "audio: no output device, sound disabled");
        return;
    }
    rt_log(LOG_INFO, "audio: output %d Hz stereo", OUT_RATE);
    std::vector<std::vector<int16_t>> bufs(NBUF, std::vector<int16_t>(BLOCK * 2));
    std::vector<WAVEHDR> hdr(NBUF);
    for (int i = 0; i < NBUF; i++) {
        hdr[i] = WAVEHDR{};
        hdr[i].lpData = (LPSTR)bufs[i].data(); hdr[i].dwBufferLength = BLOCK * 4;
        waveOutPrepareHeader(h, &hdr[i], sizeof(WAVEHDR));
        mix(bufs[i].data(), BLOCK);
        waveOutWrite(h, &hdr[i], sizeof(WAVEHDR));
    }
    for (;;) {
        WaitForSingleObject(ev, INFINITE);
        for (int i = 0; i < NBUF; i++) {
            if (!(hdr[i].dwFlags & WHDR_DONE)) continue;
            mix(bufs[i].data(), BLOCK);
            hdr[i].dwFlags &= ~WHDR_DONE;
            waveOutWrite(h, &hdr[i], sizeof(WAVEHDR));
        }
    }
}

void audio_out_init() {
    if (g_audio_wav && (g_wav = fopen(g_audio_wav, "wb"))) wav_header();
    std::thread(out_thread).detach();
}
