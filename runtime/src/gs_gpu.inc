// ================================================================ GPU renderer (Direct3D 11)
// Included by gs.cpp (it uses the GS state, VRAM helpers and texture decoder there). Selected
// with -gpu or NPPL_GPU=1; NPPL_SCALE=<n> sets the internal resolution multiplier (default 2).
//
// Model:
//  * Every frame buffer the game draws into becomes a render target (RT) at scale x the PS2 size,
//    RGBA16F: colour stored as c/255, alpha as a/128 so PS2 blend factors (0x80 = 1.0) work
//    directly as blend factors (DEST_ALPHA) and through dual-source blending (SRC1_ALPHA).
//  * Z buffers are D32_FLOAT depth buffers (PS2 Z / 2^bits; greater = nearer).
//  * Textures are decoded from VRAM by the software texture cache and uploaded, except when a
//    texture lies inside a render target in the same format: then the RT itself is sampled
//    (post-processing stays at the high resolution).
//  * VRAM stays the reference for everything the CPU side reads: before CLUT loads, texture
//    decodes and local copies touch an RT's pages its contents are read back (downsampled) into
//    VRAM; uploads into an RT's pages mark it stale and it is re-uploaded before its next use.
//  * All D3D calls happen on the GS thread (or the EE thread with NPPL_MTGS=0).
#include <d3d11.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#include <unordered_map>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#include <wincodec.h>
#include <condition_variable>
#include <deque>
#include <unordered_set>
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")

HWND display_hwnd();
bool display_shot_wanted();
void display_set_frame(const uint32_t *bgra, int w, int h, int stride);

namespace gpu {

static int S = 2;                                   // internal resolution scale
static const int RTW = 1024, RTH = 1024;             // native size of every RT / depth buffer
static ID3D11Device *dev;
static ID3D11DeviceContext *ctx;
static IDXGISwapChain1 *swap;
static ID3D11RenderTargetView *bb_rtv;
static int bb_w, bb_h;
static bool failed;

template <class T> static void rel(T *&p) { if (p) { p->Release(); p = nullptr; } }

// ---------------------------------------------------------------- render targets
struct RT {
    uint32_t fbp, fbw, psm;
    uint32_t h = 0;          // rows owned by the GPU (0..h-1)
    bool dirty = false;      // GPU contents newer than VRAM
    bool stale = false;      // VRAM newer than GPU (re-upload rows 0..h-1 before use)
    uint64_t used = 0;
    ID3D11Texture2D *tex = nullptr;
    ID3D11RenderTargetView *rtv = nullptr;
    ID3D11ShaderResourceView *srv = nullptr;
};
struct DS { uint32_t zbp; ID3D11Texture2D *tex = nullptr; ID3D11DepthStencilView *dsv = nullptr; uint64_t used = 0; };
static std::vector<RT *> rts;
static std::vector<DS *> dss;
static uint64_t frame_no;
static uint64_t dirty_gen = 1;   // bumped whenever the set of GPU-dirty pages may have grown

static uint32_t rows_per_page(uint32_t psm) { return (psm == CT16 || psm == CT16S) ? 64 : 32; }
// pages [first, end) covered by rows 0..h-1 of a buffer (PS2 pages are laid out row-major)
static void page_range(uint32_t bp, uint32_t bw, uint32_t psm, uint32_t h, uint32_t &first, uint32_t &end) {
    first = bp / 32;
    uint32_t rp = rows_per_page(psm);
    end = first + ((h + rp - 1) / rp) * std::max(bw, 1u);
}
static bool overlaps(const RT *r, uint32_t first, uint32_t end) {
    if (!r->h) return false;
    uint32_t a, b; page_range(r->fbp, r->fbw, r->psm, r->h, a, b);
    return a < end && first < b;
}

// ---------------------------------------------------------------- shaders
static const char *k_hlsl = R"(
cbuffer CB : register(b0) {
    float4 fogc;    // fog colour (0..255)
    float4 tsz;     // tw, th, RT source scale, alpha fix (0..255)
    int4 reg;       // region clamp/repeat: minu, maxu, minv, maxv
    float4 rto;     // RT source offset x, y; 1 / RT source size in pixels
    float4 misc;    // aref, TEXA ta0, TEXA aem, 0
    float4 vxf;     // ndc transform: x scale, y scale, x offset, y offset
};
Texture2D<float4> T : register(t0);
SamplerState SL : register(s0);
struct VSIn { float4 p : POSITION; float4 c : COLOR; float4 t : TEXCOORD; };
struct PSIn { float4 p : SV_Position; noperspective float4 c : COLOR; noperspective float4 t : TEXCOORD; };
PSIn vs_main(VSIn i) {
    PSIn o;
    o.p = float4(i.p.x * vxf.x + vxf.z, i.p.y * vxf.y + vxf.w, i.p.z, 1);
    o.c = i.c; o.t = i.t;
    return o;
}
#ifndef TME
#define TME 0
#endif
#ifndef TFX
#define TFX 0
#endif
#ifndef TCC
#define TCC 0
#endif
#ifndef FGE
#define FGE 0
#endif
#ifndef FST
#define FST 0
#endif
#ifndef BIL
#define BIL 0
#endif
#ifndef RTSRC
#define RTSRC 0
#endif
#ifndef WMS
#define WMS 0
#endif
#ifndef WMT
#define WMT 0
#endif
#ifndef AINV
#define AINV 0
#endif
#ifndef FBA
#define FBA 0
#endif
#ifndef IS16
#define IS16 0
#endif
#ifndef BLEND
#define BLEND 0
#endif
#ifndef CSEL
#define CSEL 0
#endif
#ifndef BA
#define BA 0
#endif
#ifndef BB
#define BB 0
#endif
#ifndef BD
#define BD 0
#endif
#ifndef PABE
#define PABE 0
#endif
#ifndef CLAMP
#define CLAMP 0
#endif
#ifndef T24
#define T24 0
#endif
#ifndef ATST
#define ATST 1
#endif
int wrapc(int c, int mode, int size, int mn, int mx) {
    if (mode == 0) return c & (size - 1);
    if (mode == 1) return clamp(c, 0, size - 1);
    if (mode == 2) { c = clamp(c, mn, mx); return c & (size - 1); }
    return ((c & mn) | mx) & (size - 1);
}
// u, v: texel coordinates in the (possibly upscaled, factor K = misc.w) texture
float4 fetch(int u, int v) {
    int tw = (int)tsz.x, th = (int)tsz.y, K = max((int)misc.w, 1);
    int nu = u >= 0 ? u / K : -((K - 1 - u) / K), nv = v >= 0 ? v / K : -((K - 1 - v) / K);
    int su = u - nu * K, sv = v - nv * K;
    return round(T.Load(int3(wrapc(nu, WMS, tw, reg.x, reg.y) * K + su, wrapc(nv, WMT, th, reg.z, reg.w) * K + sv, 0)) * 255);
}
float4 texsample(float u, float v) {
#if RTSRC
    float2 hp = float2(u + rto.x, v + rto.y) * tsz.z;
#if BIL
    float4 c = T.SampleLevel(SL, hp * rto.zw, 0);
#else
    float4 c = T.Load(int3(int2(floor(hp)), 0));
#endif
    c.rgb = round(saturate(c.rgb) * 255);
    c.a = round(clamp(c.a * 128, 0, 255));
#if T24
    c.a = (misc.z != 0 && all(c.rgb == 0)) ? 0 : misc.y;
#endif
    return c;
#elif BIL
    float KK = max(misc.w, 1);
    u = u * KK - 0.5; v = v * KK - 0.5;
    float fu0 = floor(u), fv0 = floor(v);
    int u0 = (int)fu0, v0 = (int)fv0;
    int fu = (int)((u - fu0) * 256), fv = (int)((v - fv0) * 256);
    int4 A = (int4)fetch(u0, v0), B = (int4)fetch(u0 + 1, v0), C = (int4)fetch(u0, v0 + 1), D = (int4)fetch(u0 + 1, v0 + 1);
    int4 top = A * (256 - fu) + B * fu, bot = C * (256 - fu) + D * fu;
    return (float4)((top * (256 - fv) + bot * fv + 32768) >> 16);
#else
    float KK = max(misc.w, 1);
    return fetch((int)floor(u * KK), (int)floor(v * KK));
#endif
}
struct PSOut { float4 c0 : SV_Target0; float4 c1 : SV_Target1; };
PSOut ps_main(PSIn i) {
    float4 vc = floor(i.c);
    float4 c = vc;
#if TME
    float u, v;
#if FST
    u = i.t.x; v = i.t.y;
#else
    float q = i.t.z != 0 ? i.t.z : 1e-6;
    u = i.t.x / q * tsz.x; v = i.t.y / q * tsz.y;
#endif
    float4 t = texsample(u, v);
#if TFX == 0
    c.rgb = floor(t.rgb * vc.rgb / 128); c.a = TCC ? floor(t.a * vc.a / 128) : vc.a;
#elif TFX == 1
    c.rgb = t.rgb; c.a = TCC ? t.a : vc.a;
#elif TFX == 2
    c.rgb = floor(t.rgb * vc.rgb / 128) + vc.a; c.a = TCC ? t.a + vc.a : vc.a;
#else
    c.rgb = floor(t.rgb * vc.rgb / 128) + vc.a; c.a = TCC ? t.a : vc.a;
#endif
    c = min(c, 255);
#endif
#if FGE
    float f = clamp(floor(i.t.w), 0, 255);
    c.rgb = floor((f * c.rgb + (255 - f) * fogc.rgb) / 256);
#endif
#if ATST != 1
    bool ok;
    float ar = misc.x;
#if ATST == 0
    ok = false;
#elif ATST == 2
    ok = c.a < ar;
#elif ATST == 3
    ok = c.a <= ar;
#elif ATST == 4
    ok = c.a == ar;
#elif ATST == 5
    ok = c.a >= ar;
#elif ATST == 6
    ok = c.a > ar;
#else
    ok = c.a != ar;
#endif
#if AINV
    ok = !ok;
#endif
    if (!ok) discard;
#endif
    float a = clamp(c.a, 0, 255);
    float3 rgb = c.rgb;
    PSOut o;
    o.c1 = float4(0, 0, 0, 0);
#if BLEND == 1
    // dual source: out = Cs * fs + Cd * fd, C = As or FIX
    float C = (CSEL == 0 ? c.a : tsz.w) / 128;
    float fs = (BA == 0 ? C : 0) - (BB == 0 ? C : 0) + (BD == 0 ? 1 : 0);
    float fd = (BA == 1 ? C : 0) - (BB == 1 ? C : 0) + (BD == 1 ? 1 : 0);
#if PABE
    if (c.a < 128) { fs = 1; fd = 0; }
#endif
    rgb *= fs;
    o.c1.a = fd;
#elif BLEND == 0
#if CLAMP
    rgb = clamp(rgb, 0, 255);
#else
    rgb = rgb - 256 * floor(rgb / 256);
#endif
#if IS16
    rgb = floor(rgb / 8) * 8;
#endif
#endif
#if FBA
    if (a < 128) a += 128;
#endif
#if IS16
    a = a >= 128 ? 128 : 0;
#endif
    o.c0 = float4(rgb / 255, a / 128);
    return o;
}
// full-screen helpers (SV_VertexID triangle)
struct FSOut { float4 p : SV_Position; };
FSOut vs_full(uint id : SV_VertexID) {
    FSOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.p = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}
// VRAM upload into an RT: one native texel per scale x scale block. Source alpha is 0..255.
float4 ps_upload(FSOut i) : SV_Target {
    int2 n = int2(floor(i.p.xy / tsz.z));
    float4 c = T.Load(int3(n - int2(rto.xy), 0));
    return float4(c.rgb, c.a * 255.0 / 128.0);
}
// texture upscaling: 2x edge-directed (xBR) from T (native size tsz.xy, repeating at the edges)
float4 xl(int2 c) { int2 sz = int2(tsz.xy); return T.Load(int3(((c % sz) + sz) % sz, 0)); }
float xd(float4 a, float4 b) {
    float3 d = a.rgb - b.rgb;
    float y = dot(d, float3(0.299, 0.587, 0.114)), cu = dot(d, float3(-0.169, -0.331, 0.5)), cv = dot(d, float3(0.5, -0.419, -0.081));
    return 48 * abs(y) + 7 * abs(cu) + 6 * abs(cv) + 24 * abs(a.a - b.a);
}
float4 ps_xbr(FSOut i) : SV_Target {
    int2 p = int2(i.p.xy), e = p >> 1;
    int2 s = int2((p.x & 1) ? 1 : -1, (p.y & 1) ? 1 : -1);
#define XL(dx, dy) xl(e + int2(dx, dy) * s)
    float4 E = XL(0, 0), F = XL(1, 0), H = XL(0, 1), I = XL(1, 1), D = XL(-1, 0), B = XL(0, -1);
    float4 C = XL(1, -1), G = XL(-1, 1), F4 = XL(2, 0), I4 = XL(2, 1), H5 = XL(0, 2), I5 = XL(1, 2);
    float we = xd(E, C) + xd(E, G) + xd(I, F4) + xd(I, H5) + 4 * xd(H, F);
    float wi = xd(H, D) + xd(H, I5) + xd(F, I4) + xd(F, B) + 4 * xd(E, I);
    if (we < wi) {
        float4 px = xd(E, F) <= xd(E, H) ? F : H;
        return lerp(E, px, 0.5);
    }
    return E;
}
// present: region of an RT (or VRAM readout) stretched to the window
float4 ps_present(FSOut i) : SV_Target {
    float2 uv = (i.p.xy - vxf.zw) / vxf.xy;   // 0..1 across the output rectangle
    float2 src = rto.xy + uv * tsz.xy;        // source pixels
    float4 c = T.SampleLevel(SL, src * rto.zw, 0);
    return float4(saturate(c.rgb), 1);
}
)";

static ID3D11VertexShader *vs_draw, *vs_fs;
static ID3D11PixelShader *ps_up, *ps_pres, *ps_xbr;
static int TK = 2;   // texture upscale factor (1, 2 or 4)
static ID3D11InputLayout *layout;
static ID3D11Buffer *cbuf, *vbuf;
static const UINT VB_BYTES = 16u << 20;
static UINT vb_pos;
static ID3D11SamplerState *smp_lin, *smp_pt;
static std::unordered_map<uint64_t, ID3D11PixelShader *> ps_cache;
static std::unordered_map<uint64_t, ID3D11BlendState *> blend_cache;
static std::unordered_map<uint32_t, ID3D11DepthStencilState *> depth_cache;
static ID3D11RasterizerState *rast;

static ID3DBlob *compile(const char *entry, const char *target, const std::vector<std::pair<std::string, std::string>> &defs) {
    std::vector<D3D_SHADER_MACRO> m;
    for (auto &d : defs) m.push_back({d.first.c_str(), d.second.c_str()});
    m.push_back({nullptr, nullptr});
    ID3DBlob *code = nullptr, *err = nullptr;
    HRESULT hr = D3DCompile(k_hlsl, strlen(k_hlsl), "gs_gpu", m.data(), nullptr, entry, target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &err);
    if (FAILED(hr)) {
        rt_log(LOG_ERR, "GPU: shader %s failed: %s", entry, err ? (const char *)err->GetBufferPointer() : "?");
        rel(err);
        return nullptr;
    }
    rel(err);
    return code;
}

// pixel shader key
struct PsKey {
    uint32_t tme : 1, tfx : 2, tcc : 1, fge : 1, fst : 1, bil : 1, rtsrc : 1, wms : 2, wmt : 2, atst : 3, ainv : 1,
        fba : 1, is16 : 1, blend : 2, csel : 2, ba : 2, bb : 2, bd : 2, pabe : 1, clamp : 1, t24 : 1;
};
static uint64_t ps_key_bits(const PsKey &k) { uint64_t v = 0; memcpy(&v, &k, sizeof k); return v; }
static ID3D11PixelShader *get_ps(const PsKey &k) {
    uint64_t key = ps_key_bits(k);
    auto it = ps_cache.find(key);
    if (it != ps_cache.end()) return it->second;
    auto I = [](uint32_t v) { return std::to_string(v); };
    std::vector<std::pair<std::string, std::string>> d = {
        {"TME", I(k.tme)}, {"TFX", I(k.tfx)}, {"TCC", I(k.tcc)}, {"FGE", I(k.fge)}, {"FST", I(k.fst)}, {"BIL", I(k.bil)},
        {"RTSRC", I(k.rtsrc)}, {"WMS", I(k.wms)}, {"WMT", I(k.wmt)}, {"ATST", I(k.atst)}, {"AINV", I(k.ainv)}, {"FBA", I(k.fba)},
        {"IS16", I(k.is16)}, {"BLEND", I(k.blend)}, {"CSEL", I(k.csel)}, {"BA", I(k.ba)}, {"BB", I(k.bb)}, {"BD", I(k.bd)},
        {"PABE", I(k.pabe)}, {"CLAMP", I(k.clamp)}, {"T24", I(k.t24)}};
    ID3D11PixelShader *ps = nullptr;
    if (ID3DBlob *b = compile("ps_main", "ps_5_0", d)) {
        dev->CreatePixelShader(b->GetBufferPointer(), b->GetBufferSize(), nullptr, &ps);
        rel(b);
    }
    ps_cache[key] = ps;
    return ps;
}

namespace pack { static void init(); }
// ---------------------------------------------------------------- device
static bool init() {
    if (dev || failed) return dev != nullptr;
    const char *sc = getenv("NPPL_SCALE");
    S = sc ? std::clamp(atoi(sc), 1, 8) : 2;
    UINT flags = getenv("NPPL_D3DDEBUG") ? D3D11_CREATE_DEVICE_DEBUG : 0;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0, got;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, &fl, 1, D3D11_SDK_VERSION, &dev, &got, &ctx))) {
        rt_log(LOG_ERR, "GPU: no Direct3D 11 device, using the software renderer");
        failed = true; g_gpu = false;
        return false;
    }
    {   // Input lag: by default the driver lets the CPU queue up to 3 frames ahead of the GPU,
        // which on a slow GPU is ~50+ ms between a key press and seeing it. Keep it to 1.
        // NPPL_FRAMELAT=<1-16> overrides.
        IDXGIDevice1 *xd1 = nullptr;
        const char *fl_env = getenv("NPPL_FRAMELAT");
        UINT lat = fl_env ? (UINT)std::clamp(atoi(fl_env), 1, 16) : 1;
        if (SUCCEEDED(dev->QueryInterface(__uuidof(IDXGIDevice1), (void **)&xd1))) { xd1->SetMaximumFrameLatency(lat); rel(xd1); }
    }
    ID3DBlob *v = compile("vs_main", "vs_5_0", {}), *vf = compile("vs_full", "vs_5_0", {});
    ID3DBlob *pu = compile("ps_upload", "ps_5_0", {{"WMS", "0"}, {"WMT", "0"}}), *pp = compile("ps_present", "ps_5_0", {{"WMS", "0"}, {"WMT", "0"}});
    ID3DBlob *px = compile("ps_xbr", "ps_5_0", {});
    if (!v || !vf || !pu || !pp || !px) { failed = true; g_gpu = false; return false; }
    dev->CreatePixelShader(px->GetBufferPointer(), px->GetBufferSize(), nullptr, &ps_xbr);
    rel(px);
    const char *tk = getenv("NPPL_TEXSCALE");
    TK = tk ? atoi(tk) : 2;
    TK = TK >= 4 ? 4 : TK >= 2 ? 2 : 1;
    dev->CreateVertexShader(v->GetBufferPointer(), v->GetBufferSize(), nullptr, &vs_draw);
    dev->CreateVertexShader(vf->GetBufferPointer(), vf->GetBufferSize(), nullptr, &vs_fs);
    dev->CreatePixelShader(pu->GetBufferPointer(), pu->GetBufferSize(), nullptr, &ps_up);
    dev->CreatePixelShader(pp->GetBufferPointer(), pp->GetBufferSize(), nullptr, &ps_pres);
    D3D11_INPUT_ELEMENT_DESC ie[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 32, D3D11_INPUT_PER_VERTEX_DATA, 0}};
    dev->CreateInputLayout(ie, 3, v->GetBufferPointer(), v->GetBufferSize(), &layout);
    rel(v); rel(vf); rel(pu); rel(pp);
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = 256; bd.Usage = D3D11_USAGE_DEFAULT; bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    dev->CreateBuffer(&bd, nullptr, &cbuf);
    bd.ByteWidth = VB_BYTES; bd.Usage = D3D11_USAGE_DYNAMIC; bd.BindFlags = D3D11_BIND_VERTEX_BUFFER; bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    dev->CreateBuffer(&bd, nullptr, &vbuf);
    D3D11_SAMPLER_DESC sd{};
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    dev->CreateSamplerState(&sd, &smp_lin);
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    dev->CreateSamplerState(&sd, &smp_pt);
    D3D11_RASTERIZER_DESC rd{};
    rd.FillMode = D3D11_FILL_SOLID; rd.CullMode = D3D11_CULL_NONE; rd.ScissorEnable = TRUE; rd.DepthClipEnable = FALSE;
    dev->CreateRasterizerState(&rd, &rast);
    {
        IDXGIDevice *xd = nullptr; IDXGIAdapter *ad = nullptr; DXGI_ADAPTER_DESC desc{};
        if (SUCCEEDED(dev->QueryInterface(__uuidof(IDXGIDevice), (void **)&xd)) && SUCCEEDED(xd->GetAdapter(&ad))) ad->GetDesc(&desc);
        rel(ad); rel(xd);
        char name[128]; WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof name, nullptr, nullptr);
        rt_log(LOG_INFO, "GPU: Direct3D 11 renderer on %s, %dx internal resolution, %dx textures", name, S, TK);
    }
    pack::init();
    return true;
}

static void set_cb(const void *data) { ctx->UpdateSubresource(cbuf, 0, nullptr, data, 0, 0); }

static RT *make_rt(uint32_t fbp, uint32_t fbw, uint32_t psm) {
    RT *r = new RT{fbp, fbw, psm};
    D3D11_TEXTURE2D_DESC td{};
    td.Width = RTW * S; td.Height = RTH * S; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    dev->CreateTexture2D(&td, nullptr, &r->tex);
    dev->CreateRenderTargetView(r->tex, nullptr, &r->rtv);
    dev->CreateShaderResourceView(r->tex, nullptr, &r->srv);
    rts.push_back(r);
    return r;
}
static void free_rt(RT *r) {
    rel(r->srv); rel(r->rtv); rel(r->tex);
    rts.erase(std::find(rts.begin(), rts.end(), r));
    delete r;
}

// ---------------------------------------------------------------- batching
struct Vtx { float x, y, z, w, r, g, b, a, s, t, q, f; };
struct CB {
    float fogc[4], tsz[4]; int32_t reg[4]; float rto[4], misc[4], vxf[4];
};
struct Batch {
    RT *rt = nullptr; DS *ds = nullptr;
    ID3D11ShaderResourceView *srv = nullptr;
    bool lin = false;
    PsKey ps{};
    uint32_t topo = 0;          // 0 triangles, 1 lines
    // fixed-function state
    uint32_t blend = 0, src = 0, dst = 0, op = 0;   // blend: 0 off, 1 dual source, 2 explicit factors
    uint32_t zte = 0, ztst = 0, zwrite = 0, cmask = 15;
    uint32_t afail = 0;         // 0: single pass; else second pass (inverted alpha test) with afail rules
    int sc[4] = {0, 0, 0, 0};
    CB cb{};
    bool same(const Batch &o) const {
        return rt == o.rt && ds == o.ds && srv == o.srv && lin == o.lin && ps_key_bits(ps) == ps_key_bits(o.ps) && topo == o.topo &&
               blend == o.blend && src == o.src && dst == o.dst && op == o.op && zte == o.zte && ztst == o.ztst &&
               zwrite == o.zwrite && cmask == o.cmask && afail == o.afail && !memcmp(sc, o.sc, sizeof sc) && !memcmp(&cb, &o.cb, sizeof cb);
    }
};
static Batch cur;
static std::vector<Vtx> verts;

static ID3D11BlendState *get_blend(uint32_t mode, uint32_t src, uint32_t dst, uint32_t op, uint32_t cmask) {
    uint64_t key = mode | (src << 4) | (dst << 12) | (op << 20) | ((uint64_t)cmask << 28);
    auto it = blend_cache.find(key);
    if (it != blend_cache.end()) return it->second;
    D3D11_BLEND_DESC bd{};
    auto &t = bd.RenderTarget[0];
    t.RenderTargetWriteMask = (UINT8)cmask;
    if (mode) {
        t.BlendEnable = TRUE;
        t.SrcBlend = mode == 1 ? D3D11_BLEND_ONE : (D3D11_BLEND)src;
        t.DestBlend = mode == 1 ? D3D11_BLEND_SRC1_ALPHA : (D3D11_BLEND)dst;
        t.BlendOp = mode == 1 ? D3D11_BLEND_OP_ADD : (D3D11_BLEND_OP)op;
        t.SrcBlendAlpha = D3D11_BLEND_ONE; t.DestBlendAlpha = D3D11_BLEND_ZERO; t.BlendOpAlpha = D3D11_BLEND_OP_ADD;
    }
    ID3D11BlendState *s = nullptr;
    dev->CreateBlendState(&bd, &s);
    blend_cache[key] = s;
    return s;
}
static ID3D11DepthStencilState *get_depth(uint32_t zte, uint32_t ztst, uint32_t zwrite) {
    uint32_t key = zte | (ztst << 1) | (zwrite << 3);
    auto it = depth_cache.find(key);
    if (it != depth_cache.end()) return it->second;
    D3D11_DEPTH_STENCIL_DESC dd{};
    dd.DepthEnable = zte ? TRUE : FALSE;
    dd.DepthWriteMask = zwrite ? D3D11_DEPTH_WRITE_MASK_ALL : D3D11_DEPTH_WRITE_MASK_ZERO;
    static const D3D11_COMPARISON_FUNC f[4] = {D3D11_COMPARISON_NEVER, D3D11_COMPARISON_ALWAYS, D3D11_COMPARISON_GREATER_EQUAL, D3D11_COMPARISON_GREATER};
    dd.DepthFunc = f[ztst & 3];
    ID3D11DepthStencilState *s = nullptr;
    dev->CreateDepthStencilState(&dd, &s);
    depth_cache[key] = s;
    return s;
}

static uint64_t g_flushes, g_draws;
static void flush() {
    if (verts.empty()) return;
    Batch &b = cur;
    UINT bytes = (UINT)(verts.size() * sizeof(Vtx));
    if (bytes > VB_BYTES) { verts.clear(); return; }
    D3D11_MAPPED_SUBRESOURCE m;
    D3D11_MAP mode = D3D11_MAP_WRITE_NO_OVERWRITE;
    if (vb_pos + bytes > VB_BYTES) { vb_pos = 0; mode = D3D11_MAP_WRITE_DISCARD; }
    if (FAILED(ctx->Map(vbuf, 0, mode, 0, &m))) { verts.clear(); return; }
    memcpy((uint8_t *)m.pData + vb_pos, verts.data(), bytes);
    ctx->Unmap(vbuf, 0);
    UINT stride = sizeof(Vtx), off = vb_pos;
    vb_pos += bytes;
    ctx->IASetVertexBuffers(0, 1, &vbuf, &stride, &off);
    ctx->IASetInputLayout(layout);
    ctx->IASetPrimitiveTopology(b.topo ? D3D11_PRIMITIVE_TOPOLOGY_LINELIST : D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->OMSetRenderTargets(1, &b.rt->rtv, b.ds ? b.ds->dsv : nullptr);
    D3D11_VIEWPORT vp{0, 0, (float)(RTW * S), (float)(RTH * S), 0, 1};
    ctx->RSSetViewports(1, &vp);
    D3D11_RECT r{b.sc[0], b.sc[1], b.sc[2], b.sc[3]};
    ctx->RSSetScissorRects(1, &r);
    ctx->RSSetState(rast);
    ctx->VSSetShader(vs_draw, nullptr, 0);
    set_cb(&b.cb);
    ctx->VSSetConstantBuffers(0, 1, &cbuf);
    ctx->PSSetConstantBuffers(0, 1, &cbuf);
    ctx->PSSetShaderResources(0, 1, &b.srv);
    ctx->PSSetSamplers(0, 1, b.lin ? &smp_lin : &smp_pt);
    const UINT n = (UINT)verts.size();
    ctx->PSSetShader(get_ps(b.ps), nullptr, 0);
    ctx->OMSetBlendState(get_blend(b.blend, b.src, b.dst, b.op, b.cmask), nullptr, 0xFFFFFFFF);
    ctx->OMSetDepthStencilState(get_depth(b.zte, b.ztst, b.zwrite), 0);
    ctx->Draw(n, 0);
    if (b.afail) {   // second pass: pixels failing the alpha test, with the AFAIL write rules
        PsKey k2 = b.ps; k2.ainv = 1;
        uint32_t cm = b.cmask, zw = b.zwrite;
        switch (b.afail) {
        case 1: zw = 0; break;                       // FB_ONLY
        case 2: cm = 0; break;                       // ZB_ONLY
        default: zw = 0; cm &= 7; break;             // RGB_ONLY
        }
        ctx->PSSetShader(get_ps(k2), nullptr, 0);
        ctx->OMSetBlendState(get_blend(b.blend, b.src, b.dst, b.op, cm), nullptr, 0xFFFFFFFF);
        ctx->OMSetDepthStencilState(get_depth(b.zte, b.ztst, zw), 0);
        ctx->Draw(n, 0);
    }
    ID3D11ShaderResourceView *nul = nullptr;
    ctx->PSSetShaderResources(0, 1, &nul);
    g_flushes++;
    verts.clear();
}

// ---------------------------------------------------------------- VRAM <-> RT
static ID3D11Texture2D *up_tex;   // RGBA8 native-size scratch for uploads
static ID3D11ShaderResourceView *up_srv;
static void upload_rows(RT *r, uint32_t y0, uint32_t y1) {   // VRAM rows [y0, y1) -> RT
    if (y0 >= y1) return;
    flush();
    uint32_t w = std::min(std::max(r->fbw, 1u) * 64, (uint32_t)RTW), h = y1 - y0;
    if (!up_tex) {
        D3D11_TEXTURE2D_DESC td{};
        td.Width = RTW; td.Height = RTH; td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        dev->CreateTexture2D(&td, nullptr, &up_tex);
        dev->CreateShaderResourceView(up_tex, nullptr, &up_srv);
    }
    std::vector<uint32_t> px((size_t)w * h);
    const bool is16 = r->psm == CT16 || r->psm == CT16S;
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) {
            uint32_t v = read_raw(r->psm, r->fbp, r->fbw, x, y0 + y);
            if (is16) v = ((v & 31) << 3) | (((v >> 5) & 31) << 11) | (((v >> 10) & 31) << 19) | ((v & 0x8000) ? 0x80000000u : 0);
            else if (r->psm == CT24) v &= 0xFFFFFF;
            px[(size_t)y * w + x] = v;
        }
    D3D11_BOX box{0, 0, 0, w, h, 1};
    ctx->UpdateSubresource(up_tex, 0, &box, px.data(), w * 4, 0);
    CB cb{};
    cb.tsz[2] = (float)S; cb.rto[0] = 0; cb.rto[1] = (float)y0;   // native = floor(pos / S), texel = native - (0, y0)
    set_cb(&cb);
    ctx->OMSetRenderTargets(1, &r->rtv, nullptr);
    D3D11_VIEWPORT vp{0, 0, (float)(RTW * S), (float)(RTH * S), 0, 1};
    ctx->RSSetViewports(1, &vp);
    D3D11_RECT sr{0, (LONG)(y0 * S), (LONG)(w * S), (LONG)(y1 * S)};
    ctx->RSSetScissorRects(1, &sr);
    ctx->RSSetState(rast);
    ctx->IASetInputLayout(nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(vs_fs, nullptr, 0);
    ctx->PSSetShader(ps_up, nullptr, 0);
    ctx->PSSetConstantBuffers(0, 1, &cbuf);
    ctx->PSSetShaderResources(0, 1, &up_srv);
    ctx->OMSetBlendState(get_blend(0, 0, 0, 0, 15), nullptr, 0xFFFFFFFF);
    ctx->OMSetDepthStencilState(get_depth(0, 1, 0), 0);
    ctx->Draw(3, 0);
    ID3D11ShaderResourceView *nul = nullptr;
    ctx->PSSetShaderResources(0, 1, &nul);
}

static inline float half_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023, f;
    if (e == 0) { if (!m) f = s << 31; else { float v = (float)m / 1024.0f / 16384.0f; return s ? -v : v; } }
    else if (e == 31) f = (s << 31) | 0x7F800000 | (m << 13);
    else f = (s << 31) | ((e + 112) << 23) | (m << 13);
    float out; memcpy(&out, &f, 4); return out;
}
static uint64_t g_readbacks;
// RT -> VRAM (rows 0..h-1, one sample per native pixel)
static void readback(RT *r) {
    if (!r->dirty || !r->h) return;
    flush();
    uint32_t w = std::min(std::max(r->fbw, 1u) * 64, (uint32_t)RTW), h = r->h;
    D3D11_TEXTURE2D_DESC td{};
    td.Width = w * S; td.Height = h * S; td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_STAGING; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Texture2D *st = nullptr;
    if (FAILED(dev->CreateTexture2D(&td, nullptr, &st))) return;
    D3D11_BOX box{0, 0, 0, w * S, h * S, 1};
    ctx->CopySubresourceRegion(st, 0, 0, 0, 0, r->tex, 0, &box);
    D3D11_MAPPED_SUBRESOURCE m;
    if (SUCCEEDED(ctx->Map(st, 0, D3D11_MAP_READ, 0, &m))) {
        for (uint32_t y = 0; y < h; y++) {
            const uint16_t *row = (const uint16_t *)((const uint8_t *)m.pData + (size_t)y * S * m.RowPitch);
            for (uint32_t x = 0; x < w; x++) {
                const uint16_t *p = row + (size_t)x * S * 4;
                auto cc = [](float v) { return (uint32_t)std::clamp((int)std::lround(v * 255.0f), 0, 255); };
                uint32_t cr = cc(half_to_float(p[0])), cg = cc(half_to_float(p[1])), cb = cc(half_to_float(p[2]));
                uint32_t ca = (uint32_t)std::clamp((int)std::lround(half_to_float(p[3]) * 128.0f), 0, 255);
                uint32_t v;
                if (r->psm == CT16 || r->psm == CT16S) v = (cr >> 3) | ((cg >> 3) << 5) | ((cb >> 3) << 10) | ((ca & 0x80) ? 0x8000 : 0);
                else v = cr | (cg << 8) | (cb << 16) | (ca << 24);
                write_raw(r->psm, r->fbp, r->fbw, x, y, v);
            }
        }
        ctx->Unmap(st, 0);
    }
    rel(st);
    r->dirty = false;
    g_write_gen++;
    g_readbacks++;
}

// The CPU is about to read VRAM pages [first, end): bring dirty RTs there up to date.
static void sync_read(uint32_t first, uint32_t end) {
    for (RT *r : rts) if (r->dirty && overlaps(r, first, end)) readback(r);
}
// The CPU is about to write pages [first, end): RTs there must re-upload afterwards.
static void sync_write(uint32_t first, uint32_t end) {
    for (RT *r : rts)
        if (overlaps(r, first, end)) { readback(r); r->stale = true; }
}
static void sync_rect_pages(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t y0, uint32_t h, bool write) {
    uint32_t a, b; page_range(bp, bw, psm, y0 + h, a, b);
    if (write) sync_write(a, b); else sync_read(a, b);
}

// Render target for a draw (creating / resizing / refreshing it as needed)
static RT *get_rt(uint32_t fbp, uint32_t fbw, uint32_t psm, uint32_t need_h) {
    uint32_t cls = (psm == CT16 || psm == CT16S) ? psm : CT32;
    need_h = std::min(need_h, (uint32_t)RTH);
    RT *r = nullptr;
    for (RT *x : rts) if (x->fbp == fbp) { r = x; break; }
    if (r && (r->fbw != fbw || (((r->psm == CT16 || r->psm == CT16S) ? r->psm : CT32) != cls))) { readback(r); free_rt(r); r = nullptr; }
    if (!r) {
        // drop RTs that overlap the new one
        uint32_t a, b; page_range(fbp, fbw, psm, need_h, a, b);
        for (size_t i = 0; i < rts.size();) {
            if (overlaps(rts[i], a, b)) { readback(rts[i]); free_rt(rts[i]); } else i++;
        }
        r = make_rt(fbp, fbw, psm);
        r->stale = false;
    }
    r->psm = psm;
    if (r->stale) { upload_rows(r, 0, r->h); r->stale = false; }
    if (need_h > r->h) {
        uint32_t old = r->h;
        // the new rows may overlap other RTs (packed buffers): drop those first
        uint32_t a, b; page_range(fbp, fbw, psm, need_h, a, b);
        for (size_t i = 0; i < rts.size();) {
            if (rts[i] != r && overlaps(rts[i], a, b)) { readback(rts[i]); free_rt(rts[i]); } else i++;
        }
        upload_rows(r, old, need_h);
        r->h = need_h;
        dirty_gen++;
    }
    r->used = frame_no;
    return r;
}
static DS *get_ds(uint32_t zbp) {
    for (DS *d : dss) if (d->zbp == zbp) { d->used = frame_no; return d; }
    DS *d = new DS{zbp};
    D3D11_TEXTURE2D_DESC td{};
    td.Width = RTW * S; td.Height = RTH * S; td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_D32_FLOAT;
    td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    dev->CreateTexture2D(&td, nullptr, &d->tex);
    dev->CreateDepthStencilView(d->tex, nullptr, &d->dsv);
    flush();
    ctx->ClearDepthStencilView(d->dsv, D3D11_CLEAR_DEPTH, 0.0f, 0);
    dss.push_back(d);
    d->used = frame_no;
    return d;
}

// ---------------------------------------------------------------- texture packs
// Every decoded texture has a content hash (pixels + size). With -texdump (NPPL_TEXDUMP=1) new
// textures are written to texdump/<hash>.png; a PNG named <hash>*.png in textures/ (an upscaled
// copy, any integer multiple of the original size) replaces the texture when it is loaded.
// PNG alpha is the PS2 alpha doubled (0x80 = opaque -> 255).
namespace pack {
static std::unordered_map<uint64_t, std::wstring> files;   // hash -> replacement file
static bool dump;
static IWICImagingFactory *wic;
static std::mutex mx;
static std::condition_variable cv;
static std::deque<std::pair<uint64_t, std::vector<uint32_t>>> q;   // dump queue (hash, w|h<<16 header + pixels)
static std::unordered_set<uint64_t> dumped;
static uint64_t n_loaded, n_dumped;

static uint64_t hash_tex(const uint32_t *p, uint32_t w, uint32_t h) {
    uint64_t x = 0xCBF29CE484222325ull ^ ((uint64_t)w << 32 | h);
    size_t n = (size_t)w * h;
    for (size_t i = 0; i < n; i++) { x ^= p[i]; x *= 0x100000001B3ull; }
    x ^= x >> 31; x *= 0x9E3779B97F4A7C15ull; x ^= x >> 29;
    return x;
}
static bool ensure_wic() {
    if (wic) return true;
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    return SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic)));
}
static void dump_worker() {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    for (;;) {
        std::pair<uint64_t, std::vector<uint32_t>> job;
        {
            std::unique_lock<std::mutex> lk(mx);
            cv.wait(lk, [] { return !q.empty(); });
            job = std::move(q.front()); q.pop_front();
        }
        uint32_t w = job.second[0] & 0xFFFF, h = job.second[0] >> 16;
        std::vector<uint8_t> bgra((size_t)w * h * 4);
        for (size_t i = 0; i < (size_t)w * h; i++) {
            uint32_t c = job.second[1 + i];
            bgra[i * 4 + 0] = (uint8_t)(c >> 16); bgra[i * 4 + 1] = (uint8_t)(c >> 8); bgra[i * 4 + 2] = (uint8_t)c;
            bgra[i * 4 + 3] = (uint8_t)std::min<uint32_t>((c >> 24) * 2, 255);
        }
        wchar_t name[64];
        swprintf(name, 64, L"texdump\\%016llx.png", (unsigned long long)job.first);
        IWICStream *st = nullptr; IWICBitmapEncoder *enc = nullptr; IWICBitmapFrameEncode *fr = nullptr;
        bool ok = SUCCEEDED(wic->CreateStream(&st)) && SUCCEEDED(st->InitializeFromFilename(name, GENERIC_WRITE)) &&
                  SUCCEEDED(wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc)) && SUCCEEDED(enc->Initialize(st, WICBitmapEncoderNoCache)) &&
                  SUCCEEDED(enc->CreateNewFrame(&fr, nullptr)) && SUCCEEDED(fr->Initialize(nullptr)) && SUCCEEDED(fr->SetSize(w, h));
        if (ok) {
            WICPixelFormatGUID pf = GUID_WICPixelFormat32bppBGRA;
            ok = SUCCEEDED(fr->SetPixelFormat(&pf)) && pf == GUID_WICPixelFormat32bppBGRA &&
                 SUCCEEDED(fr->WritePixels(h, w * 4, (UINT)bgra.size(), bgra.data())) && SUCCEEDED(fr->Commit()) && SUCCEEDED(enc->Commit());
        }
        rel(fr); rel(enc); rel(st);
        if (ok) n_dumped++;
    }
}
static void init() {
    dump = getenv("NPPL_TEXDUMP") && atoi(getenv("NPPL_TEXDUMP")) != 0;
    WIN32_FIND_DATAW fd;
    HANDLE f = FindFirstFileW(L"textures\\*.png", &fd);
    if (f != INVALID_HANDLE_VALUE) {
        do {
            uint64_t h = 0;
            if (swscanf(fd.cFileName, L"%16llx", (unsigned long long *)&h) == 1) files[h] = std::wstring(L"textures\\") + fd.cFileName;
        } while (FindNextFileW(f, &fd));
        FindClose(f);
    }
    if (!files.empty() || dump) ensure_wic();
    if (dump) {
        CreateDirectoryA("texdump", nullptr);
        WIN32_FIND_DATAW d2;
        HANDLE g = FindFirstFileW(L"texdump\\*.png", &d2);   // keep earlier dumps
        if (g != INVALID_HANDLE_VALUE) {
            do { uint64_t h; if (swscanf(d2.cFileName, L"%16llx", (unsigned long long *)&h) == 1) dumped.insert(h); } while (FindNextFileW(g, &d2));
            FindClose(g);
        }
        std::thread(dump_worker).detach();
    }
    rt_log(LOG_INFO, "GPU: texture pack: %zu replacement textures%s", files.size(), dump ? ", dumping new textures to texdump/" : "");
}
static void queue_dump(uint64_t h, const uint32_t *p, uint32_t w, uint32_t hh) {
    if (!dump || !wic || w > 0xFFFF || hh > 0xFFFF || !dumped.insert(h).second) return;
    std::vector<uint32_t> v(1 + (size_t)w * hh);
    v[0] = w | (hh << 16);
    memcpy(v.data() + 1, p, (size_t)w * hh * 4);
    { std::lock_guard<std::mutex> lk(mx); q.emplace_back(h, std::move(v)); }
    cv.notify_one();
}
// Load the replacement for texture `h` (w x h native); k = scale factor. Empty when none / invalid.
static bool load(uint64_t h, uint32_t w, uint32_t hh, std::vector<uint32_t> &out, int &k) {
    auto it = files.find(h);
    static int dbg;
    if (getenv("NPPL_PACKLOG") && dbg++ < 20) rt_log(LOG_INFO, "pack: lookup %016llx %ux%u -> %s (wic %p)", (unsigned long long)h, w, hh, it == files.end() ? "miss" : "hit", (void *)wic);
    if (it == files.end() || !wic) return false;
    IWICBitmapDecoder *dec = nullptr; IWICBitmapFrameDecode *fr = nullptr; IWICBitmapSource *cv32 = nullptr;
    bool ok = false;
    UINT W = 0, H = 0;
    if (SUCCEEDED(wic->CreateDecoderFromFilename(it->second.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec)) &&
        SUCCEEDED(dec->GetFrame(0, &fr)) && SUCCEEDED(WICConvertBitmapSource(GUID_WICPixelFormat32bppRGBA, fr, &cv32)) &&
        SUCCEEDED(cv32->GetSize(&W, &H)) && W % w == 0 && H % hh == 0 && W / w == H / hh && W / w >= 1 && W / w <= 16) {
        out.resize((size_t)W * H);
        ok = SUCCEEDED(cv32->CopyPixels(nullptr, W * 4, W * H * 4, (BYTE *)out.data()));
        if (ok) for (uint32_t &c : out) c = (c & 0xFFFFFF) | ((((c >> 24) + 1) >> 1) << 24);   // alpha back to PS2 scale
        k = (int)(W / w);
    }
    rel(cv32); rel(fr); rel(dec);
    if (!ok) { files.erase(it); rt_log(LOG_WARN, "GPU: texture pack: %016llx.png has the wrong size, ignored", (unsigned long long)h); }
    return ok;
}
}   // namespace pack

// ---------------------------------------------------------------- textures
struct GTex { ID3D11Texture2D *tex = nullptr; ID3D11ShaderResourceView *srv = nullptr; uint64_t used = 0; int k = 1; };
// One 2x xBR pass: src (w x h) -> new texture (2w x 2h)
static ID3D11Texture2D *xbr_pass(ID3D11ShaderResourceView *src, uint32_t w, uint32_t h, ID3D11ShaderResourceView **srv_out) {
    D3D11_TEXTURE2D_DESC td{};
    td.Width = w * 2; td.Height = h * 2; td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    ID3D11Texture2D *t = nullptr;
    ID3D11RenderTargetView *rtv = nullptr;
    if (FAILED(dev->CreateTexture2D(&td, nullptr, &t))) return nullptr;
    dev->CreateRenderTargetView(t, nullptr, &rtv);
    dev->CreateShaderResourceView(t, nullptr, srv_out);
    CB cb{};
    cb.tsz[0] = (float)w; cb.tsz[1] = (float)h;
    set_cb(&cb);
    ctx->OMSetRenderTargets(1, &rtv, nullptr);
    D3D11_VIEWPORT vp{0, 0, (float)(w * 2), (float)(h * 2), 0, 1};
    ctx->RSSetViewports(1, &vp);
    D3D11_RECT sr{0, 0, (LONG)(w * 2), (LONG)(h * 2)};
    ctx->RSSetScissorRects(1, &sr);
    ctx->RSSetState(rast);
    ctx->IASetInputLayout(nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(vs_fs, nullptr, 0);
    ctx->PSSetShader(ps_xbr, nullptr, 0);
    ctx->PSSetConstantBuffers(0, 1, &cbuf);
    ctx->PSSetShaderResources(0, 1, &src);
    ctx->OMSetBlendState(get_blend(0, 0, 0, 0, 15), nullptr, 0xFFFFFFFF);
    ctx->OMSetDepthStencilState(get_depth(0, 1, 0), 0);
    ctx->Draw(3, 0);
    ID3D11ShaderResourceView *nul = nullptr;
    ctx->PSSetShaderResources(0, 1, &nul);
    ID3D11RenderTargetView *nrt = nullptr;
    ctx->OMSetRenderTargets(1, &nrt, nullptr);
    rel(rtv);
    return t;
}
static uint64_t g_tex_uploads;
static std::unordered_map<uint64_t, GTex> tex_cache;
static uint32_t tex_epoch_seen;
static ID3D11ShaderResourceView *upload_tex(const TexState &t, int &k_out) {
    if (tex_epoch_seen != g_tex_epoch) {
        flush();
        for (auto &e : tex_cache) { rel(e.second.srv); rel(e.second.tex); }
        tex_cache.clear(); tex_epoch_seen = g_tex_epoch;
    }
    uint64_t key = t.uid ^ ((uint64_t)(uintptr_t)t.pix * 0x9E3779B97F4A7C15ull) ^ ((uint64_t)t.tw << 48) ^ ((uint64_t)t.th << 56);
    auto it = tex_cache.find(key);
    if (it != tex_cache.end()) { it->second.used = frame_no; k_out = it->second.k; return it->second.srv; }
    if (tex_cache.size() > 3000) {   // evict textures unused for a while
        flush();
        for (auto i = tex_cache.begin(); i != tex_cache.end();)
            if (i->second.used + 30 < frame_no) { rel(i->second.srv); rel(i->second.tex); i = tex_cache.erase(i); } else ++i;
    }
    GTex g;
    D3D11_TEXTURE2D_DESC td{};
    td.Width = t.tw; td.Height = t.th; td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_IMMUTABLE; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    if (pack::dump || !pack::files.empty()) {
        uint64_t h = pack::hash_tex(t.pix, t.tw, t.th);
        std::vector<uint32_t> rp;
        int rk = 1;
        if (pack::load(h, t.tw, t.th, rp, rk)) {   // replacement texture from the pack
            td.Width = t.tw * rk; td.Height = t.th * rk;
            D3D11_SUBRESOURCE_DATA rd{rp.data(), td.Width * 4, 0};
            if (SUCCEEDED(dev->CreateTexture2D(&td, &rd, &g.tex))) {
                dev->CreateShaderResourceView(g.tex, nullptr, &g.srv);
                g.k = rk; g.used = frame_no;
                pack::n_loaded++; g_tex_uploads++;
                tex_cache[key] = g;
                k_out = g.k;
                return g.srv;
            }
            td.Width = t.tw; td.Height = t.th;
        }
        pack::queue_dump(h, t.pix, t.tw, t.th);
    }
    D3D11_SUBRESOURCE_DATA sd{t.pix, t.tw * 4, 0};
    dev->CreateTexture2D(&td, &sd, &g.tex);
    dev->CreateShaderResourceView(g.tex, nullptr, &g.srv);
    // upscale (the native copy is dropped; views keep their textures alive)
    uint32_t w = t.tw, h = t.th;
    while (g.k < TK && w * 2 <= 8192 && h * 2 <= 8192) {
        ID3D11ShaderResourceView *nsrv = nullptr;
        ID3D11Texture2D *nt = xbr_pass(g.srv, w, h, &nsrv);
        if (!nt) break;
        rel(g.srv); rel(g.tex);
        g.tex = nt; g.srv = nsrv; g.k *= 2; w *= 2; h *= 2;
    }
    g.used = frame_no;
    g_tex_uploads++;
    tex_cache[key] = g;
    k_out = g.k;
    return g.srv;
}
// Copy of an RT for draws that sample the target they render into
static ID3D11Texture2D *fb_copy;
static ID3D11ShaderResourceView *fb_copy_srv;
static ID3D11ShaderResourceView *copy_of(RT *r) {
    flush();
    if (!fb_copy) {
        D3D11_TEXTURE2D_DESC td{};
        r->tex->GetDesc(&td);
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        dev->CreateTexture2D(&td, nullptr, &fb_copy);
        dev->CreateShaderResourceView(fb_copy, nullptr, &fb_copy_srv);
    }
    ctx->CopyResource(fb_copy, r->tex);
    return fb_copy_srv;
}

// ---------------------------------------------------------------- draw
static void emit(const Vertex &v, float depth, const Vertex &col, bool fst) {
    Vtx o;
    o.x = v.x; o.y = v.y; o.z = depth; o.w = 1;
    o.r = (float)col.r; o.g = (float)col.g; o.b = (float)col.b; o.a = (float)col.a;
    if (fst) { o.s = v.u; o.t = v.v; o.q = 1; } else { o.s = v.s; o.t = v.t; o.q = v.q; }
    o.f = (float)v.fog;
    verts.push_back(o);
}

static uint64_t g_unsupported_logged;
static void draw(DrawCtx &d, const Vertex *v, int nv) {
    if (!init()) return;
    const uint32_t prim = d.prim;
    // target
    float maxy = 0;
    for (int i = 0; i < nv; i++) maxy = std::max(maxy, v[i].y);
    uint32_t need_h = (uint32_t)std::clamp(std::min((int)std::ceil(maxy) + 1, d.scy1 + 1), 1, RTH);
    Batch b;
    b.rt = get_rt(d.fbp, d.fbw, d.fpsm, need_h);
    if (!b.rt->dirty) { b.rt->dirty = true; dirty_gen++; }
    // depth
    b.zte = d.zte; b.ztst = d.ztst; b.zwrite = d.zte && !d.zmsk;
    if (d.zte) b.ds = get_ds(d.zbp);
    // texture
    PsKey &k = b.ps;
    k = PsKey{};
    if (d.tme) {
        TexState &t = d.tex;
        k.tme = 1; k.tfx = t.tfx & 3; k.tcc = t.tcc & 1; k.fst = d.fst; k.bil = t.bilinear; k.wms = t.wms & 3; k.wmt = t.wmt & 3;
        // inside a render target with a compatible format? sample it directly
        RT *src = nullptr;
        uint32_t ox = 0, oy = 0;
        bool t32 = t.psm == CT32 || t.psm == CT24, t16 = t.psm == CT16 || t.psm == CT16S;
        if (t32 || t16)
            for (RT *r : rts) {
                bool r32 = r->psm == CT32 || r->psm == CT24;
                if (!r->h || (t32 != r32) || (t16 && t.psm != r->psm) || t.tbw != r->fbw || t.tbp < r->fbp || (t.tbp - r->fbp) % 32) continue;
                uint32_t pg = (t.tbp - r->fbp) / 32, rp = rows_per_page(r->psm);
                uint32_t py = (pg / std::max(r->fbw, 1u)) * rp, px = (pg % std::max(r->fbw, 1u)) * 64;
                if (py >= r->h) continue;
                src = r; ox = px; oy = py;
                break;
            }
        if (src) {
            if (src->stale) { upload_rows(src, 0, src->h); src->stale = false; }
            k.rtsrc = 1; k.t24 = t.psm == CT24;
            b.srv = src == b.rt ? copy_of(src) : src->srv;
            b.lin = t.bilinear;
            b.cb.tsz[2] = (float)S;
            b.cb.rto[0] = (float)ox; b.cb.rto[1] = (float)oy;
            b.cb.rto[2] = 1.0f / (float)(RTW * S); b.cb.rto[3] = 1.0f / (float)(RTH * S);
            b.cb.misc[1] = (float)bits(t.texa, 0, 8); b.cb.misc[2] = (float)bits(t.texa, 15, 1);
        } else {
            // texture pages the GPU may have drawn: bring VRAM up to date (memoized per texture)
            static struct { uint32_t tbp, tbw, psm, tw, th; uint64_t gen; } last{};
            if (!(last.gen == dirty_gen && last.tbp == t.tbp && last.tbw == t.tbw && last.psm == t.psm && last.tw == t.tw && last.th == t.th)) {
                gpu_sync_rect(t.psm, t.tbp, t.tbw, 0, 0, t.tw, t.th, false);
                last = {t.tbp, t.tbw, t.psm, t.tw, t.th, dirty_gen};
            }
            t.nlev = 1;
            tex_bind(t);
            if (!t.pix) return;
            int kk = 1;
            b.srv = upload_tex(t, kk);
            b.cb.misc[3] = (float)kk;
            b.cb.reg[0] = (int)t.minu; b.cb.reg[1] = (int)t.maxu; b.cb.reg[2] = (int)t.minv; b.cb.reg[3] = (int)t.maxv;
        }
        b.cb.tsz[0] = (float)t.tw; b.cb.tsz[1] = (float)t.th;
    }
    k.fge = d.fge;
    if (d.fge) { b.cb.fogc[0] = (float)d.fogr; b.cb.fogc[1] = (float)d.fogg; b.cb.fogc[2] = (float)d.fogb; }
    // alpha test
    k.atst = d.ate ? d.atst & 7 : 1;
    b.cb.misc[0] = (float)d.aref;
    if (d.ate && d.atst != 1 && d.afail) b.afail = d.afail;
    if (d.date && !(g_unsupported_logged & 1)) { g_unsupported_logged |= 1; rt_log(LOG_INFO, "GPU: destination alpha test not implemented yet"); }
    k.fba = d.fba; k.is16 = d.fpsm == CT16 || d.fpsm == CT16S; k.clamp = d.colclamp;
    // blending
    if (d.abe) {
        k.ba = d.aa & 3; k.bb = d.ab & 3; k.bd = d.ad & 3; k.csel = d.ac & 3; k.pabe = d.pabe;
        b.cb.tsz[3] = (float)d.afix;
        if (d.ac != 1) {
            k.blend = 1; b.blend = 1;
        } else {
            // C = destination alpha: fixed-function factors; express fs, fd as m + kk * Ad
            auto coef = [&](int which, int &m, int &kk) { kk = (d.aa == (uint32_t)which) - (d.ab == (uint32_t)which); m = d.ad == (uint32_t)which; };
            int ms, ks, md, kd; coef(0, ms, ks); coef(1, md, kd);
            auto fac = [](int m, int kk, int &f) {
                if (kk == 0) { f = m ? D3D11_BLEND_ONE : D3D11_BLEND_ZERO; return true; }
                if (kk == 1 && !m) { f = D3D11_BLEND_DEST_ALPHA; return true; }
                if (kk == -1 && m) { f = D3D11_BLEND_INV_DEST_ALPHA; return true; }
                return false;
            };
            int fs, fd;
            k.blend = 2; b.blend = 2; b.op = D3D11_BLEND_OP_ADD;
            if (fac(ms, ks, fs) && fac(md, kd, fd)) { b.src = fs; b.dst = fd; }
            else if (ks == -1 && !ms && fac(md, kd, fd)) { b.src = D3D11_BLEND_DEST_ALPHA; b.dst = fd; b.op = D3D11_BLEND_OP_REV_SUBTRACT; }
            else if (kd == -1 && !md && fac(ms, ks, fs)) { b.src = fs; b.dst = D3D11_BLEND_DEST_ALPHA; b.op = D3D11_BLEND_OP_SUBTRACT; }
            else {
                b.src = D3D11_BLEND_ONE; b.dst = D3D11_BLEND_ZERO;
                if (!(g_unsupported_logged & 2)) { g_unsupported_logged |= 2; rt_log(LOG_INFO, "GPU: blend mode %u%u%u%u approximated", d.aa, d.ab, d.ac, d.ad); }
            }
        }
    }
    // colour write mask (whole bytes only)
    uint32_t msk = d.fbmsk, cm = 0;
    if ((msk & 0xFF) != 0xFF) cm |= 1;
    if ((msk & 0xFF00) != 0xFF00) cm |= 2;
    if ((msk & 0xFF0000) != 0xFF0000) cm |= 4;
    if ((msk & 0xFF000000u) != 0xFF000000u && d.fpsm != CT24) cm |= 8;
    b.cmask = cm;
    // scissor (high-res pixels), viewport transform
    b.sc[0] = d.scx0 * S; b.sc[1] = d.scy0 * S; b.sc[2] = (d.scx1 + 1) * S; b.sc[3] = (d.scy1 + 1) * S;
    const float W = (float)(RTW * S), H = (float)(RTH * S);
    b.cb.vxf[0] = 2.0f * S / W; b.cb.vxf[1] = -2.0f * S / H; b.cb.vxf[2] = 1.0f / W - 1.0f; b.cb.vxf[3] = 1.0f - 1.0f / H;
    b.topo = (prim == 1 || prim == 2) ? 1 : 0;
    if (!(cur.rt && cur.same(b))) { flush(); cur = b; }
    // depth scale
    const double zs = d.zpsm == Z32 ? 1.0 / 4294967296.0 : (d.zpsm == Z24 ? 1.0 / 16777216.0 : 1.0 / 65536.0);
    auto dep = [&](uint32_t z) { uint32_t zm = zmax(d.zpsm); return (float)((double)std::min(z, zm) * zs); };
    const bool fst = d.fst;
    switch (prim) {
    case 0: {   // point: a one-pixel quad
        Vertex a = v[0], q[4] = {a, a, a, a};
        q[1].x += 1; q[2].y += 1; q[3].x += 1; q[3].y += 1;
        float z = dep(a.z);
        emit(q[0], z, a, fst); emit(q[1], z, a, fst); emit(q[2], z, a, fst);
        emit(q[1], z, a, fst); emit(q[3], z, a, fst); emit(q[2], z, a, fst);
        break;
    }
    case 1: case 2: {
        const Vertex &c0 = d.iip ? v[0] : v[1];
        emit(v[0], dep(v[0].z), c0, fst); emit(v[1], dep(v[1].z), v[1], fst);
        break;
    }
    case 3: case 4: case 5: {
        for (int i = 0; i < 3; i++) emit(v[i], dep(v[i].z), d.iip ? v[i] : v[2], fst);
        break;
    }
    case 6: {   // sprite: two triangles, flat colour / Z / Q from the second vertex
        const Vertex &a = v[0], &c = v[1];
        Vertex p[4];
        for (int i = 0; i < 4; i++) {
            Vertex q = c;
            bool rx = i & 1, ry = i & 2;
            q.x = rx ? c.x : a.x; q.y = ry ? c.y : a.y;
            q.u = rx ? c.u : a.u; q.v = ry ? c.v : a.v;
            q.s = rx ? c.s : a.s; q.t = ry ? c.t : a.t;
            q.q = c.q;
            p[i] = q;
        }
        float z = dep(c.z);
        emit(p[0], z, c, fst); emit(p[1], z, c, fst); emit(p[2], z, c, fst);
        emit(p[1], z, c, fst); emit(p[3], z, c, fst); emit(p[2], z, c, fst);
        break;
    }
    }
    g_draws++;
    if (verts.size() * sizeof(Vtx) > VB_BYTES / 2) flush();
}

// ---------------------------------------------------------------- present
static ID3D11Texture2D *vram_disp;
static ID3D11ShaderResourceView *vram_disp_srv;
static void ensure_swap() {
    HWND h = display_hwnd();
    if (!h) return;
    RECT rc; GetClientRect(h, &rc);
    int w = std::max(1, (int)rc.right), hh = std::max(1, (int)rc.bottom);
    if (swap && (w != bb_w || hh != bb_h)) {
        rel(bb_rtv);
        swap->ResizeBuffers(0, (UINT)w, (UINT)hh, DXGI_FORMAT_UNKNOWN, 0);
        bb_w = w; bb_h = hh;
    }
    if (!swap) {
        IDXGIDevice *xd = nullptr; IDXGIAdapter *ad = nullptr; IDXGIFactory2 *fac = nullptr;
        dev->QueryInterface(__uuidof(IDXGIDevice), (void **)&xd);
        xd->GetAdapter(&ad);
        ad->GetParent(__uuidof(IDXGIFactory2), (void **)&fac);
        DXGI_SWAP_CHAIN_DESC1 sd{};
        sd.Width = (UINT)w; sd.Height = (UINT)hh; sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM; sd.SampleDesc.Count = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; sd.BufferCount = 2; sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        if (FAILED(fac->CreateSwapChainForHwnd(dev, h, &sd, nullptr, nullptr, &swap))) rt_log(LOG_ERR, "GPU: swap chain creation failed");
        fac->MakeWindowAssociation(h, DXGI_MWA_NO_ALT_ENTER);
        rel(fac); rel(ad); rel(xd);
        bb_w = w; bb_h = hh;
    }
    if (swap && !bb_rtv) {
        ID3D11Texture2D *bb = nullptr;
        swap->GetBuffer(0, __uuidof(ID3D11Texture2D), (void **)&bb);
        dev->CreateRenderTargetView(bb, nullptr, &bb_rtv);
        rel(bb);
    }
}
static void shot(ID3D11ShaderResourceView *srv, ID3D11Texture2D *srctex, float sx, float sy, float sw, float sh, bool fp16) {
    // debug screenshot of the displayed region at internal resolution
    int w = (int)sw, h = (int)sh;
    D3D11_TEXTURE2D_DESC td{};
    srctex->GetDesc(&td);
    td.Width = (UINT)w; td.Height = (UINT)h; td.BindFlags = 0; td.Usage = D3D11_USAGE_STAGING; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ; td.MiscFlags = 0;
    ID3D11Texture2D *st = nullptr;
    if (FAILED(dev->CreateTexture2D(&td, nullptr, &st))) return;
    D3D11_BOX box{(UINT)sx, (UINT)sy, 0, (UINT)(sx + w), (UINT)(sy + h), 1};
    ctx->CopySubresourceRegion(st, 0, 0, 0, 0, srctex, 0, &box);
    D3D11_MAPPED_SUBRESOURCE m;
    if (SUCCEEDED(ctx->Map(st, 0, D3D11_MAP_READ, 0, &m))) {
        std::vector<uint32_t> out((size_t)w * h);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                uint32_t c;
                if (fp16) {
                    const uint16_t *p = (const uint16_t *)((const uint8_t *)m.pData + (size_t)y * m.RowPitch) + x * 4;
                    auto cc = [](float v) { return (uint32_t)std::clamp((int)std::lround(v * 255.0f), 0, 255); };
                    c = (cc(half_to_float(p[0])) << 16) | (cc(half_to_float(p[1])) << 8) | cc(half_to_float(p[2]));
                } else {
                    uint32_t v = ((const uint32_t *)((const uint8_t *)m.pData + (size_t)y * m.RowPitch))[x];
                    c = ((v & 255) << 16) | (v & 0xFF00) | ((v >> 16) & 255);
                }
                out[(size_t)y * w + x] = c;
            }
        ctx->Unmap(st, 0);
        display_set_frame(out.data(), w, h, w);
    }
    rel(st);
    (void)srv;
}
static void present() {
    if (!init()) return;
    flush();
    frame_no++;
    // stats (every 3 s)
    static auto t0 = std::chrono::steady_clock::now();
    if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(3)) {
        t0 = std::chrono::steady_clock::now();
        if (pack::dump || pack::n_loaded) rt_log(LOG_INFO, "GPU: texture pack: %llu replaced, %llu dumped", (unsigned long long)pack::n_loaded, (unsigned long long)pack::n_dumped);
        rt_log(LOG_INFO, "GPU: %llu draws, %llu batches, %llu readbacks, %llu texture uploads, %zu RTs, %zu textures, %zu shaders",
               (unsigned long long)g_draws, (unsigned long long)g_flushes, (unsigned long long)g_readbacks, (unsigned long long)g_tex_uploads,
               rts.size(), tex_cache.size(), ps_cache.size());
        g_draws = g_flushes = g_readbacks = g_tex_uploads = 0;
    }
    ensure_swap();
    uint64_t pmode = g_priv[0], dispfb = g_priv[0x90 >> 3], display = g_priv[0xA0 >> 3];
    if (!(pmode & 2) && (pmode & 1)) { dispfb = g_priv[0x70 >> 3]; display = g_priv[0x80 >> 3]; }
    uint32_t fbp = (uint32_t)bits(dispfb, 0, 9) * 32, fbw = (uint32_t)bits(dispfb, 9, 6), psm = (uint32_t)bits(dispfb, 15, 5);
    uint32_t dbx = (uint32_t)bits(dispfb, 32, 11), dby = (uint32_t)bits(dispfb, 43, 11);
    uint32_t magh = (uint32_t)bits(display, 23, 4) + 1, magv = (uint32_t)bits(display, 27, 2) + 1;
    uint32_t dw = (uint32_t)bits(display, 32, 12) + 1, dh = (uint32_t)bits(display, 44, 11) + 1;
    int w = (int)(dw / magh), h = (int)(dh / magv);
    if (h > 512) h /= 2;
    if (w <= 0 || h <= 0 || !fbw) return;
    RT *r = nullptr;
    for (RT *x : rts) if (x->fbp == fbp && x->h) { r = x; break; }
    ID3D11ShaderResourceView *srv;
    ID3D11Texture2D *srctex;
    float sx, sy, sw, sh, tw, th;
    bool fp16;
    if (r) {
        if (r->stale) { upload_rows(r, 0, r->h); r->stale = false; }
        srv = r->srv; srctex = r->tex; fp16 = true;
        sx = (float)(dbx * S); sy = (float)(dby * S); sw = (float)(w * S); sh = (float)(h * S);
        tw = (float)(RTW * S); th = (float)(RTH * S);
    } else {
        // not drawn by the GPU (e.g. an uploaded image): show VRAM
        static std::vector<uint32_t> px(1024 * 1024);
        int rw, rh;
        sync_rect_pages(psm, fbp, fbw, dby, (uint32_t)h, false);
        if (!gs_read_display(px.data(), 1024, 1024, rw, rh)) return;
        if (!vram_disp) {
            D3D11_TEXTURE2D_DESC td{};
            td.Width = 1024; td.Height = 1024; td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            dev->CreateTexture2D(&td, nullptr, &vram_disp);
            dev->CreateShaderResourceView(vram_disp, nullptr, &vram_disp_srv);
        }
        ctx->UpdateSubresource(vram_disp, 0, nullptr, px.data(), 1024 * 4, 0);
        srv = vram_disp_srv; srctex = vram_disp; fp16 = false;
        sx = 0; sy = 0; sw = (float)rw; sh = (float)rh; tw = 1024; th = 1024;
    }
    if (display_shot_wanted()) shot(srv, srctex, sx, sy, sw, sh, fp16);
    if (!swap || !bb_rtv) return;
    CB cb{};
    cb.tsz[0] = sw; cb.tsz[1] = sh;
    cb.rto[0] = sx; cb.rto[1] = sy; cb.rto[2] = 1.0f / tw; cb.rto[3] = 1.0f / th;
    cb.vxf[0] = (float)bb_w; cb.vxf[1] = (float)bb_h; cb.vxf[2] = 0; cb.vxf[3] = 0;
    set_cb(&cb);
    ctx->OMSetRenderTargets(1, &bb_rtv, nullptr);
    D3D11_VIEWPORT vp{0, 0, (float)bb_w, (float)bb_h, 0, 1};
    ctx->RSSetViewports(1, &vp);
    D3D11_RECT sr{0, 0, bb_w, bb_h};
    ctx->RSSetScissorRects(1, &sr);
    ctx->RSSetState(rast);
    ctx->IASetInputLayout(nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(vs_fs, nullptr, 0);
    ctx->PSSetShader(ps_pres, nullptr, 0);
    ctx->PSSetConstantBuffers(0, 1, &cbuf);
    ctx->PSSetShaderResources(0, 1, &srv);
    ctx->PSSetSamplers(0, 1, &smp_lin);
    ctx->OMSetBlendState(get_blend(0, 0, 0, 0, 15), nullptr, 0xFFFFFFFF);
    ctx->OMSetDepthStencilState(get_depth(0, 1, 0), 0);
    ctx->Draw(3, 0);
    ID3D11ShaderResourceView *nul = nullptr;
    ctx->PSSetShaderResources(0, 1, &nul);
    swap->Present(0, 0);
}

}   // namespace gpu

// CPU access to a VRAM rectangle: read back (and for writes, mark stale) the render targets on its pages
static void gpu_sync_rect(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h, bool write) {
    using namespace gpu;
    if (!dev || rts.empty() || !w || !h) return;
    static uint8_t hit[NPAGES];
    uint32_t lo = NPAGES, hi = 0;
    memset(hit, 0, sizeof hit);
    for_rect_pages(psm, bp, bw, x0, y0, w, h, [&](uint32_t p) { hit[p] = 1; lo = std::min(lo, p); hi = std::max(hi, p); return false; });
    for (RT *r : rts) {
        if (!r->h || (!write && !r->dirty)) continue;
        uint32_t a, b; page_range(r->fbp, r->fbw, r->psm, r->h, a, b);
        bool any = false;
        for (uint32_t p = a; p < b && !any; p++) any = hit[p & (NPAGES - 1)] != 0;
        if (!any) continue;
        readback(r);
        if (write) r->stale = true;
    }
}
