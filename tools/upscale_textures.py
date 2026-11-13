#!/usr/bin/env python3
"""Build a texture pack from textures dumped by the game (nppl_pc.exe -texdump).

    python tools/upscale_textures.py <texdump dir> <textures dir> [--esrgan path\\to\\realesrgan-ncnn-vulkan.exe]

Every <hash>.png in the dump folder is upscaled 4x with Real-ESRGAN (realesrgan-ncnn-vulkan, a
separate download) and written to the output folder under the same name; the game loads
textures/<hash>.png from its working directory in place of the original texture. Textures are
padded with wrapped copies of their own edges before upscaling so tiling textures stay seamless.
Already converted textures are skipped, so the script can be re-run after more dumping.
Without --esrgan it falls back to a plain Lanczos resize (no added detail).

The dumped and upscaled textures are derived from the game's data: keep them out of git.
"""
import argparse, os, shutil, subprocess, sys, tempfile
import numpy as np
from PIL import Image

SCALE = 4


def pad_wrap(im, p):
    a = np.asarray(im)
    py, px = min(p, a.shape[0]), min(p, a.shape[1])
    return Image.fromarray(np.pad(a, ((py, py), (px, px), (0, 0)), mode='wrap')), px, py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('dst')
    ap.add_argument('--esrgan', help='path to realesrgan-ncnn-vulkan.exe')
    ap.add_argument('--model', default='realesrgan-x4plus')
    ap.add_argument('--pad', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    todo = [f for f in sorted(os.listdir(a.src)) if f.lower().endswith('.png') and not os.path.exists(os.path.join(a.dst, f))]
    print(f'{len(todo)} textures to upscale')
    if not todo:
        return
    tmp = tempfile.mkdtemp(prefix='texup_')
    tin, tout = os.path.join(tmp, 'in'), os.path.join(tmp, 'out')
    os.makedirs(tin); os.makedirs(tout)
    pads = {}
    for f in todo:
        im = Image.open(os.path.join(a.src, f)).convert('RGBA')
        pim, px, py = pad_wrap(im, a.pad)
        pads[f] = (px, py, im.width, im.height)
        pim.save(os.path.join(tin, f))
    if a.esrgan:
        a.esrgan = os.path.abspath(a.esrgan)
        cmd = [a.esrgan, '-i', tin, '-o', tout, '-n', a.model, '-s', str(SCALE), '-f', 'png']
        print(' '.join(cmd))
        r = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(a.esrgan)))
        if r.returncode:
            sys.exit('Real-ESRGAN failed')
    n = 0
    for f in todo:
        px, py, w, h = pads[f]
        src = os.path.join(tout, f)
        if a.esrgan and os.path.exists(src):
            up = Image.open(src).convert('RGBA')
        else:
            pim = Image.open(os.path.join(tin, f))
            up = pim.resize((pim.width * SCALE, pim.height * SCALE), Image.LANCZOS)
        up = up.crop((px * SCALE, py * SCALE, (px + w) * SCALE, (py + h) * SCALE))
        up.save(os.path.join(a.dst, f))
        n += 1
    shutil.rmtree(tmp, ignore_errors=True)
    print(f'wrote {n} textures to {a.dst}')


if __name__ == '__main__':
    main()
