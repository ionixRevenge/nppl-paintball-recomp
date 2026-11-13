#!/usr/bin/env python3
"""DATA.ARC tool: the archive is a sequence of zlib streams (with padding).

  arc_tool.py scan    DATA.ARC index.tsv        catalog every stream
  arc_tool.py extract DATA.ARC index.tsv outdir  decompress all entries

index.tsv columns: idx, offset, comp_size, raw_size, magic/guess
The file names are not stored in the archive; they will be recovered from the
game's loader (see FileS* / ".ARC" references near 0x711281 in the ELF).
"""
import os, sys, zlib

CHUNK = 1 << 20

def guess(head):
    s = head[:64]
    if b"BSP Compiler" in head[:256]: return "bsp"
    if s[:4] in (b"VAGp", b"pGAV"): return "vag"
    if s[:4] == b"RIFF": return "riff"
    if s[:4].isalnum(): return "magic:" + s[:4].decode()
    return "bin:" + s[:8].hex()

def scan(path, out):
    f = open(path, "rb"); size = os.path.getsize(path)
    pos, idx = 0, 0
    o = open(out, "w")
    o.write("idx\toffset\tcomp\traw\ttype\n")
    while pos < size:
        f.seek(pos)
        # skip padding to next zlib header (78 01/5e/9c/da)
        buf = f.read(CHUNK)
        j = 0
        while j < len(buf) - 1 and not (buf[j] == 0x78 and buf[j + 1] in (0x01, 0x5E, 0x9C, 0xDA)):
            j += 1
        if j >= len(buf) - 1:
            if len(buf) < CHUNK: break
            pos += len(buf) - 1; continue
        pos += j
        f.seek(pos)
        d = zlib.decompressobj()
        raw, head, consumed = 0, b"", 0
        try:
            while not d.eof:
                c = f.read(CHUNK)
                if not c: break
                r = d.decompress(c)
                if len(head) < 256: head += r[:256]
                raw += len(r)
                consumed += len(c) - len(d.unused_data)
        except zlib.error:
            pos += 1; continue  # false positive header
        if not d.eof:
            break
        o.write(f"{idx}\t{pos}\t{consumed}\t{raw}\t{guess(head)}\n")
        idx += 1; pos += consumed
        if idx % 500 == 0:
            print(f"{idx} entries, {pos * 100 // size}%", flush=True)
    print(f"done: {idx} entries")

def extract(path, index, outdir):
    os.makedirs(outdir, exist_ok=True)
    f = open(path, "rb")
    for line in list(open(index))[1:]:
        idx, off, comp, raw, typ = line.rstrip("\n").split("\t")
        f.seek(int(off))
        data = zlib.decompress(f.read(int(comp)))
        ext = typ.split(":")[0]
        open(os.path.join(outdir, f"{int(idx):05d}.{ext}"), "wb").write(data)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "scan": scan(sys.argv[2], sys.argv[3])
    elif cmd == "extract": extract(sys.argv[2], sys.argv[3], sys.argv[4])
    else: sys.exit(__doc__)
