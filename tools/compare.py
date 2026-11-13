#!/usr/bin/env python3
"""Compare a rebuilt main segment (objcopy -O binary) against the original ELF.

Reports OK, or the first mismatching functions (using build/SLUS_218.55.map
when available) so you can see which decompiled function broke matching.
"""
import bisect, re, sys

FOFF, VRAM, SIZE = 0x80, 0x100000, 0x682B80

def load_map(path="build/SLUS_218.55.map"):
    syms = []
    try:
        for line in open(path, errors="replace"):
            m = re.match(r"\s+0x0*([0-9a-fA-F]+)\s+([A-Za-z_][\w.$]*)\s*$", line)
            if m:
                syms.append((int(m.group(1), 16), m.group(2)))
    except FileNotFoundError:
        pass
    syms.sort()
    return syms

def main():
    orig = open(sys.argv[1], "rb").read()[FOFF:FOFF + SIZE]
    built = open(sys.argv[2], "rb").read()
    if built == orig:
        print("OK: main segment matches (%d bytes)" % SIZE)
        return 0
    syms = load_map()
    addrs = [s[0] for s in syms]
    if len(built) != len(orig):
        print("size mismatch: built 0x%X, original 0x%X" % (len(built), len(orig)))
    bad, last = [], None
    for i in range(0, min(len(built), len(orig)), 4):
        if built[i:i + 4] != orig[i:i + 4]:
            va = VRAM + i
            k = bisect.bisect_right(addrs, va) - 1
            name = syms[k][1] if k >= 0 else "?"
            if name != last:
                bad.append((va, name)); last = name
                if len(bad) >= 20:
                    break
    for va, name in bad:
        print("mismatch at 0x%08X in %s" % (va, name))
    return 1

if __name__ == "__main__":
    sys.exit(main())
