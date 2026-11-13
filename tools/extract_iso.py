#!/usr/bin/env python3
"""Extract all files from an ISO9660 disc image (skips TRASH/ padding by default)."""
import argparse, os, struct

def extract(iso_path, out, include_padding=False, quiet=False):
    f = open(iso_path, "rb")

    def rd(lba, n):
        f.seek(lba * 2048)
        return f.read(n)

    pvd = rd(16, 2048)
    if pvd[1:6] != b"CD001":
        raise ValueError("not an ISO9660 image")

    def walk(lba, size, path):
        data, i = rd(lba, size), 0
        while i < len(data):
            l = data[i]
            if l == 0:
                i = (i // 2048 + 1) * 2048
                continue
            rec = data[i:i + l]
            elba = struct.unpack_from("<I", rec, 2)[0]
            esz = struct.unpack_from("<I", rec, 10)[0]
            name = rec[33:33 + rec[32]]
            if name not in (b"\0", b"\1"):
                p = os.path.join(path, name.decode().split(";")[0])
                if rec[25] & 2:
                    walk(elba, esz, p)
                elif "TRASH" in p.upper() and not include_padding:
                    pass
                else:
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    f.seek(elba * 2048)
                    with open(p, "wb") as o:
                        left = esz
                        while left:
                            c = f.read(min(left, 1 << 24))
                            if not c:
                                raise IOError(f"ISO truncated while reading {p}")
                            o.write(c)
                            left -= len(c)
                    if not quiet:
                        print(f"{esz:>11} {p}")
            i += l

    root = pvd[156:190]
    walk(struct.unpack_from("<I", root, 2)[0], struct.unpack_from("<I", root, 10)[0], out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("iso")
    ap.add_argument("out")
    ap.add_argument("--include-padding", action="store_true")
    a = ap.parse_args()
    extract(a.iso, a.out, a.include_padding)

if __name__ == "__main__":
    main()
