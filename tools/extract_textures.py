#!/usr/bin/env python3
"""Export every texture in DATA.ARC as <hash>.png, the same names nppl_pc.exe -texdump writes.

    python tools/extract_textures.py <arc dir> <out dir> [--index build/data_arc_index.tsv] [--check <texdump dir>]

<arc dir> holds the archive entries unpacked by tools/arc_tool.py (NNNNN.bin). Each texture entry
(header 00 01 00 00) is decoded the way the GS sees it after upload: the 8-bit index data is stored
in GS PSMT8 block order and the 256-colour palette in CSM1 order, both are undone here. Every mip
level is exported, since the game uploads them as separate textures. The file name is the runtime
content hash (pack::hash_tex in runtime/src/gs_gpu.inc), so the output folder can be fed straight to
tools/upscale_textures.py and the results dropped in the game's textures/ folder.
Alpha is doubled in the PNG (PS2 alpha 0x80 = opaque), matching the runtime dumps.

The exported textures are game data: keep them out of git.
"""
import argparse, os, struct
import numpy as np
from PIL import Image

M64 = (1 << 64) - 1
CSM1 = np.array([(i & 0xE7) | ((i & 8) << 1) | ((i & 16) >> 1) for i in range(256)])


def hash_tex(px, w, h):
    """Must stay identical to pack::hash_tex in runtime/src/gs_gpu.inc."""
    x = 0xCBF29CE484222325 ^ ((w << 32) | h)
    for p in px.tolist():
        x = ((x ^ p) * 0x100000001B3) & M64
    x ^= x >> 31
    x = (x * 0x9E3779B97F4A7C15) & M64
    x ^= x >> 29
    return '%016x' % x


_maps = {}
def unswizzle8(w, h):
    """Index map from GS PSMT8 page layout (as uploaded through a 32-bit view) to linear order."""
    if (w, h) not in _maps:
        y, x = np.mgrid[0:h, 0:w]
        block = (y & ~0xF) * w + (x & ~0xF) * 2
        swap = (((y + 2) >> 2) & 1) * 4
        py = (((y & ~3) >> 1) + (y & 1)) & 7
        column = py * w * 2 + ((x + swap) & 7) * 4
        byte = ((y >> 1) & 1) + ((x >> 2) & 2)
        _maps[(w, h)] = (block + column + byte).ravel()
    return _maps[(w, h)]


def decode(d):
    """Yield (w, h, rgba uint32 array) for every mip level of a texture entry."""
    _, w, h, mips, flags, psm = struct.unpack_from('<6I', d, 0)
    if psm == 0x13:
        bpp = 8
    elif psm == 0x14:
        bpp = 4
    else:
        return
    sizes = [((w >> m), (h >> m)) for m in range(mips)]
    sizes = [(mw, mh) for mw, mh in sizes if mw and mh]
    off = 0x28 + sum(mw * mh * bpp // 8 for mw, mh in sizes)
    ncol = 256 if bpp == 8 else 16
    if off + ncol * 4 > len(d):
        return
    pal = np.frombuffer(d, '<u4', ncol, off)
    if bpp == 8:
        pal = pal[CSM1]
    off = 0x28
    for mw, mh in sizes:
        n = mw * mh * bpp // 8
        raw = np.frombuffer(d, np.uint8, n, off)
        off += n
        if bpp == 8:
            idx = raw[unswizzle8(mw, mh)] if mw >= 16 and mh >= 16 else raw
        else:
            idx = np.empty(mw * mh, np.uint8)
            idx[0::2] = raw & 15
            idx[1::2] = raw >> 4
        yield mw, mh, pal[idx]


def save_png(px, w, h, path):
    a = px.view(np.uint8).reshape(h, w, 4).copy()
    a[..., 3] = np.minimum(a[..., 3].astype(np.uint16) * 2, 255)
    Image.fromarray(a, 'RGBA').save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arc'); ap.add_argument('out')
    ap.add_argument('--index', help='data_arc_index.tsv (optional; otherwise every .bin is scanned)')
    ap.add_argument('--check', help='texdump folder: report how many runtime dumps were reproduced')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.index:
        rows = [l.split('\t') for l in open(a.index).read().splitlines()[1:]]
        files = ['%05d.bin' % int(r[0]) for r in rows if r[4].startswith('bin:00010000')]
    else:
        files = sorted(f for f in os.listdir(a.arc) if f.endswith('.bin'))
    seen = set(); ntex = 0
    for f in files:
        d = open(os.path.join(a.arc, f), 'rb').read()
        if d[:4] != b'\x00\x01\x00\x00' or len(d) < 0x28:
            continue
        ntex += 1
        for w, h, px in decode(d):
            if w < 8 or h < 8:
                continue
            hs = hash_tex(px, w, h)
            if hs in seen:
                continue
            seen.add(hs)
            p = os.path.join(a.out, hs + '.png')
            if not os.path.exists(p):
                save_png(px, w, h, p)
    print(f'{ntex} texture entries, {len(seen)} unique textures (incl. mips) in {a.out}')
    if a.check:
        dumped = {f[:16] for f in os.listdir(a.check) if f.endswith('.png')}
        print(f'runtime dumps reproduced: {len(dumped & seen)} / {len(dumped)}')
        missing = sorted(dumped - seen)
        if missing:
            print('not in archive textures:', ' '.join(missing[:40]))


if __name__ == '__main__':
    main()
