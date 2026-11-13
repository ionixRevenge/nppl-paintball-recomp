#!/usr/bin/env python3
"""Find string references per function (lui/addiu|ori pairs) in the main ELF.

Outputs a TSV: func_addr <tab> string1 | string2 | ...
Used to label library/game modules (Havok asserts, SCE error messages, etc).
"""
import argparse, re, struct, sys
import rabbitizer

BASE, FOFF = 0x100000, 0x80
TEXT = [(0x100000, 0x683B00), (0x743C80, 0x76C5E0)]

def load_funcs(paths):
    addrs = set()
    for p in paths:
        for m in re.finditer(r'INCLUDE_ASM\("[^"]+", (\w+)\);', open(p).read()):
            n = m.group(1)
            mm = re.match(r'func_([0-9A-F]{8})$', n)
            if mm: addrs.add(int(mm.group(1), 16))
    return sorted(addrs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elf"); ap.add_argument("out")
    ap.add_argument("--src", nargs="+", default=["src/main_text.c", "src/text2.c"])
    ap.add_argument("--symbols", default="config/symbol_addrs.txt")
    a = ap.parse_args()
    d = open(a.elf, "rb").read()[FOFF:]
    end = BASE + len(d)
    funcs = load_funcs(a.src)
    for m in re.finditer(r'^\s*\w+\s*=\s*0x([0-9A-Fa-f]+);\s*//\s*type:func', open(a.symbols).read(), re.M):
        funcs.append(int(m.group(1), 16))
    funcs = sorted(set(funcs))

    def cstr(va):
        if not (BASE <= va < end): return None
        o = va - BASE
        e = d.find(b"\0", o, o + 512)
        if e <= o + 3: return None
        s = d[o:e]
        if all(32 <= c < 127 or c in (9, 10) for c in s):
            return s.decode().replace("\n", "\n").replace("\t", "\t")
        return None

    out = open(a.out, "w", encoding="utf-8")
    for i, f in enumerate(funcs):
        nxt = funcs[i + 1] if i + 1 < len(funcs) else f + 0x4000
        seg_end = next((e for s, e in TEXT if s <= f < e), None)
        if seg_end is None: continue
        nxt = min(nxt, seg_end)
        hi = {}
        found = []
        for va in range(f, nxt, 4):
            w = struct.unpack_from("<I", d, va - BASE)[0]
            ins = rabbitizer.Instruction(w, vram=va, category=rabbitizer.InstrCategory.R5900)
            op = ins.getOpcodeName()
            if op == "lui":
                hi[ins.rt.value] = ins.getProcessedImmediate() << 16
            elif op in ("addiu", "ori", "daddiu") and ins.rs.value in hi:
                imm = ins.getProcessedImmediate()
                val = (hi[ins.rs.value] + imm) & 0xFFFFFFFF
                s = cstr(val)
                if s and s not in found: found.append(s)
        if found:
            out.write("%08X\t%s\n" % (f, " | ".join(found)))
    print("functions scanned:", len(funcs))

if __name__ == "__main__":
    main()
