#!/usr/bin/env python3
"""Static recompiler: PS2 EE (R5900) ELF -> C.

Reads the user's own executable (iso/SLUS_218.55) and emits C that runs on the
runtime in runtime/. Output goes to build/recomp/ and is NEVER committed: it is
derived from the game binary.

  python recomp/recomp.py iso/SLUS_218.55 build/recomp [--files 128]

Model:
  * every guest function -> `void f_XXXXXXXX(R5900 *c)`
  * branches -> goto, delay slots emitted in the correct order
  * jal -> direct C call, jr $ra -> return
  * any address code can reach indirectly (data pointers, lui/addiu pairs,
    out-of-function branch targets) becomes an entry: functions with several
    entries get a `switch (entry)` at the top
  * jr <reg> -> switch over in-function code pointers, else dispatch
"""
import argparse, bisect, os, re, struct, sys
from collections import defaultdict

import rabbitizer

FOFF, VRAM = 0x80, 0x100000
TEXT = [(0x100000, 0x683B00), (0x743C80, 0x76C5E0)]
DATA = [(0x683B00, 0x743C80), (0x76C5E0, 0x782B80)]

R5900 = rabbitizer.InstrCategory.R5900

def in_text(a):
    return any(s <= a < e for s, e in TEXT)

def seg_end(a):
    for s, e in TEXT:
        if s <= a < e:
            return e
    return None

# ---------------------------------------------------------------- helpers
def R(n):   return "0" if n == 0 else f"c->r[{n}].ud[0]"
def RS(n):  return "0" if n == 0 else f"(int64_t)c->r[{n}].ud[0]"
def RW(n):  return "0" if n == 0 else f"(uint32_t)c->r[{n}].ud[0]"
def RSW(n): return "0" if n == 0 else f"(int32_t)c->r[{n}].ud[0]"
def F(n):   return f"c->f[{n}].f"
def FU(n):  return f"c->f[{n}].u"
def h(x):   return "0x%08X" % (x & 0xFFFFFFFF)

def setr(n, expr):
    return "" if n == 0 else f"c->r[{n}].ud[0] = {expr};"

def sx32(expr):
    return f"SX32({expr})"

BRANCH_COND = {
    "beq":  lambda s, t: f"{R(s)} == {R(t)}",
    "bne":  lambda s, t: f"{R(s)} != {R(t)}",
    "blez": lambda s, t: f"{RS(s)} <= 0",
    "bgtz": lambda s, t: f"{RS(s)} > 0",
    "bltz": lambda s, t: f"{RS(s)} < 0",
    "bgez": lambda s, t: f"{RS(s)} >= 0",
    "bc1t": lambda s, t: "(c->fcr31 & FCR_C)",
    "bc1f": lambda s, t: "!(c->fcr31 & FCR_C)",
    # CPCOND0 (DMA idle): report "true" so wait loops on bc0f fall through
    "bc0t": lambda s, t: "1",
    "bc0f": lambda s, t: "0",
}

VU_BC = "xyzw"

# ---------------------------------------------------------------- decoder
class Insn:
    __slots__ = ("va", "w", "op", "rs", "rt", "rd", "sa", "imm", "simm", "valid", "ri")
    def __init__(self, va, w):
        self.va, self.w = va, w
        ri = rabbitizer.Instruction(w, vram=va, category=R5900)
        self.ri = ri
        self.valid = ri.isValid()
        self.op = ri.getOpcodeName() if self.valid else "INVALID"
        if self.op == "c1" and (w >> 21) & 31 == 16 and w & 63 == 4:
            self.op = "sqrt.s"
        self.rs, self.rt, self.rd = (w >> 21) & 31, (w >> 16) & 31, (w >> 11) & 31
        self.sa = (w >> 6) & 31
        self.imm = w & 0xFFFF
        self.simm = self.imm - 0x10000 if self.imm & 0x8000 else self.imm

    @property
    def base_op(self):
        return self.op[:-1] if self.op in ("beql", "bnel", "blezl", "bgtzl", "bltzl", "bgezl", "bc1tl", "bc1fl") else self.op

    def is_branch(self):
        return self.base_op in BRANCH_COND or self.op in ("b", "beqz", "bnez", "beqzl", "bnezl")

    def is_likely(self):
        return self.op in ("beql", "bnel", "blezl", "bgtzl", "bltzl", "bgezl", "bc1tl", "bc1fl", "beqzl", "bnezl")

    def branch_target(self):
        return (self.va + 4 + (self.simm << 2)) & 0xFFFFFFFF

    def jump_target(self):
        return ((self.va + 4) & 0xF0000000) | ((self.w & 0x3FFFFFF) << 2)

    def has_delay(self):
        return self.is_branch() or self.op in ("j", "jal", "jr", "jalr")


class Recomp:
    def __init__(self, elf):
        raw = open(elf, "rb").read()
        self.d = raw[FOFF:FOFF + 0x682B80]
        self.end = VRAM + len(self.d)
        self.cache = {}
        self.unhandled = defaultdict(int)
        self.delay_dup = 0

    def word(self, va):
        return struct.unpack_from("<I", self.d, va - VRAM)[0]

    def insn(self, va):
        i = self.cache.get(va)
        if i is None:
            i = self.cache[va] = Insn(va, self.word(va))
        return i

    # ------------------------------------------------------------ analysis
    def load_functions(self, srcs, symfile):
        starts = set()
        names = {}
        for p in srcs:
            for m in re.finditer(r'INCLUDE_ASM\("[^"]+", (\w+)\);', open(p).read()):
                n = m.group(1)
                mm = re.match(r"func_([0-9A-F]{8})$", n)
                if mm:
                    starts.add(int(mm.group(1), 16))
                else:
                    names[n] = None
        for m in re.finditer(r"^\s*(\w+)\s*=\s*0x([0-9A-Fa-f]+);\s*//.*type:func", open(symfile).read(), re.M):
            a = int(m.group(2), 16)
            starts.add(a)
            names[m.group(1)] = a
        self.sym = {a: n for n, a in names.items() if a is not None}
        starts.add(TEXT[0][0])
        for s, e in TEXT:
            starts.add(s)
        self.starts = sorted(a for a in starts if in_text(a))

    def load_overrides(self, path):
        self.overrides = {}
        if not path or not os.path.exists(path):
            return
        for line in open(path):
            line = line.split("#")[0].strip()
            if not line:
                continue
            a, n = line.split()[:2]
            self.overrides[int(a, 16)] = n
        print(f"HLE overrides: {len(self.overrides)}")

    def analyze(self):
        entries = set(self.starts)
        code_ptrs = set()
        # 1) code pointers stored in data
        for s, e in DATA:
            for va in range(s, e, 4):
                v = self.word(va)
                if in_text(v) and v % 4 == 0:
                    code_ptrs.add(v)
        # 2) scan code: jal targets, out-of-range branches, lui/addiu code pointers
        jal_targets, far_targets, built_ptrs = set(), set(), set()
        for fi, fs in enumerate(self.starts):
            fe = self.func_end(fi)
            hi = {}
            for va in range(fs, fe, 4):
                i = self.insn(va)
                if i.op == "jal":
                    jal_targets.add(i.jump_target())
                elif i.op == "j" or i.is_branch():
                    t = i.jump_target() if i.op == "j" else i.branch_target()
                    if not (fs <= t < fe):
                        far_targets.add(t)
                if i.op == "lui":
                    hi[i.rt] = (i.imm << 16) & 0xFFFFFFFF
                elif i.op in ("addiu", "ori", "daddiu") and i.rs in hi:
                    v = (hi[i.rs] + (i.imm if i.op == "ori" else i.simm)) & 0xFFFFFFFF
                    if in_text(v) and v % 4 == 0:
                        built_ptrs.add(v)
                elif i.op == "jal" or i.op == "jalr":
                    hi.clear()
        self.code_ptrs = code_ptrs | built_ptrs
        for t in jal_targets | far_targets | self.code_ptrs:
            if in_text(t):
                entries.add(t)
        self.entries = sorted(entries)
        # map each entry to its containing function
        self.func_entries = defaultdict(list)
        for e in self.entries:
            k = bisect.bisect_right(self.starts, e) - 1
            self.func_entries[self.starts[k]].append(e)
        print(f"functions: {len(self.starts)}, entries: {len(self.entries)} "
              f"(jal {len(jal_targets)}, data ptrs {len(code_ptrs)}, built ptrs {len(built_ptrs)}, far {len(far_targets)})")

    def func_end(self, fi):
        fs = self.starts[fi]
        nxt = self.starts[fi + 1] if fi + 1 < len(self.starts) else self.end
        return min(nxt, seg_end(fs))

    # ------------------------------------------------------------ codegen
    def fname(self, a):
        return f"f_{a:08X}"

    def call_expr(self, t):
        if t in self.entry_set:
            return f"{self.fname(t)}(c);"
        return f"rt_call(c, {h(t)});"

    def emit_func(self, fi, out):
        fs, fe = self.starts[fi], self.func_end(fi)
        if fs in self.overrides and self.overrides[fs] != "setjmp":
            n = self.overrides[fs]
            lines = [f"void hle_{n}(R5900 *c);", f"void {self.fname(fs)}(R5900 *c) {{ hle_{n}(c); }}"]
            for e in sorted(self.func_entries[fs]):
                if e != fs:
                    lines.append(f"void {self.fname(e)}(R5900 *c) {{ rt_bad_entry(c, {h(e)}); }}")
            out.append(chr(10).join(lines))
            return
        ents = sorted(self.func_entries[fs])
        labels = set(e for e in ents if e != fs)
        jr_cases = sorted(p for p in self.code_ptrs if fs <= p < fe)
        for va in range(fs, fe, 4):
            i = self.insn(va)
            if i.is_branch() or i.op == "j":
                t = i.jump_target() if i.op == "j" else i.branch_target()
                if fs <= t < fe:
                    labels.add(t)
        labels |= set(jr_cases)
        # a delay slot that is also a branch target is emitted twice (inline and
        # labelled); the inline path must then skip the labelled copy
        for va in range(fs, fe, 4):
            if va + 4 in labels and va + 8 < fe and self.insn(va).has_delay():
                labels.add(va + 8)
        self.cur = (fs, fe, jr_cases)

        multi = len(ents) > 1 or ents[0] != fs
        body = self.fname(fs) + ("_body" if multi else "")
        lines = []
        if multi:
            lines.append(f"static void {body}(R5900 *c, uint32_t entry) {{")
            lines.append("    c->pc = entry;")
            lines.append("    switch (entry) {")
            for e in ents:
                lines.append(f"    case {h(e)}: goto L_{e:08X};")
            lines.append("    default: rt_bad_entry(c, entry); return;")
            lines.append("    }")
        else:
            lines.append(f"void {body}(R5900 *c) {{")
            lines.append(f"    c->pc = {h(fs)};")
        va = fs
        last_uncond = False
        while va < fe:
            i = self.insn(va)
            if va in labels or (multi and va in ents):
                lines.append(f"L_{va:08X}:;")
            if i.has_delay():
                d = self.insn(va + 4) if va + 4 < fe else Insn(va + 4, 0)
                if va + 4 in labels:
                    # a branch target in a delay slot: emit it inline here and
                    # also as its own labelled instruction afterwards
                    pass
                self.cur_va = va
                code, last_uncond = self.emit_control(i, d)
                lines.append(f"    /* {va:08X} */ {code}")
                if va + 4 in labels:
                    self.delay_dup += 1
                    if not last_uncond:
                        skip = f"goto L_{va + 8:08X};" if va + 8 < fe else f"{self.call_expr(fe)} return;"
                        lines.append(f"    {skip} /* skip duplicated delay slot */")
                    lines.append(f"L_{va + 4:08X}:;")
                    lines.append(f"    /* {va + 4:08X} */ {self.emit_simple(d)}")
                    last_uncond = False
                va += 8
            else:
                lines.append(f"    /* {va:08X} */ {self.emit_simple(i)}")
                last_uncond = False
                va += 4
        if not last_uncond:
            lines.append(f"    {self.call_expr(fe)} return; /* fallthrough */")
        lines.append("}")
        if multi:
            for e in ents:
                lines.append(f"void {self.fname(e)}(R5900 *c) {{ {body}(c, {h(e)}); }}")
        out.append("\n".join(lines))

    def goto_or_tail(self, t):
        fs, fe, _ = self.cur
        if fs <= t < fe:
            if t <= self.cur_va:
                return f"{{ RT_BACK(c, {h(t)}); goto L_{t:08X}; }}"
            return f"goto L_{t:08X};"
        return f"{{ {self.call_expr(t)} return; }}"

    def emit_control(self, i, d):
        """Returns (code, is_unconditional_exit)."""
        ds = self.emit_simple(d) if d.valid or d.w == 0 else self.emit_simple(d)
        op = i.op
        if d.has_delay():
            ds = f"/* branch in delay slot @{d.va:08X} ignored */ " + ds
        if op == "jal":
            t = i.jump_target()
            if self.overrides.get(t) == "setjmp":
                # guest setjmp saves the registers; the host setjmp makes the matching
                # hle_longjmp resume here (with $v0 already set) instead of inside setjmp
                return (f"c->r[31].ud[0] = {h(i.va + 8)}; {ds} "
                        f"if (!setjmp(*(jmp_buf *)rt_jmpbuf((uint32_t)c->r[4].ud[0]))) {self.call_expr(t)}"), False
            return f"c->r[31].ud[0] = {h(i.va + 8)}; {ds} {self.call_expr(t)}", False
        if op == "jalr":
            rd = i.rd
            pre = f"uint32_t _t = {RW(i.rs)};"
            link = setr(rd, h(i.va + 8))
            return f"{{ {pre} {link} {ds} rt_call(c, _t); }}", False
        if op == "jr":
            if i.rs == 31:
                return f"{ds} return;", True
            fs, fe, cases = self.cur
            sw = ""
            if cases:
                sw = "switch (_t) { " + " ".join(f"case {h(x)}: goto L_{x:08X};" for x in cases) + " } "
            return f"{{ uint32_t _t = {RW(i.rs)}; {ds} {sw}rt_call(c, _t); return; }}", True
        if op == "j":
            return f"{ds} {self.goto_or_tail(i.jump_target())}", True
        if op == "b":
            return f"{ds} {self.goto_or_tail(i.branch_target())}", True
        # conditional branches
        if op in ("beqz", "beqzl"):
            cond = f"{R(i.rs)} == 0"
        elif op in ("bnez", "bnezl"):
            cond = f"{R(i.rs)} != 0"
        else:
            cond = BRANCH_COND[i.base_op](i.rs, i.rt)
        t = i.branch_target()
        if i.is_likely():
            return f"if ({cond}) {{ {ds} {self.goto_or_tail(t)} }}", False
        uncond = op == "beq" and i.rs == i.rt
        return f"{{ int _c = {cond}; {ds} if (_c) {self.goto_or_tail(t)} }}", uncond

    def emit_simple(self, i):
        op, s, t, d, sa = i.op, i.rs, i.rt, i.rd, i.sa
        imm, simm = i.imm, i.simm
        ea = f"(uint32_t)({R(s)} + {simm})" if s else h(simm)
        if op in ("nop", "sync", "sync.p", "pref", "vnop", "vwaitq", "cache", "ssnop"):
            return ";"
        # --- ALU immediate
        if op in ("addiu", "addi"):  return setr(t, sx32(f"(uint32_t){R(s)} + {simm}"))
        if op == "daddiu":           return setr(t, f"{R(s)} + (uint64_t){simm}")
        if op == "lui":              return setr(t, "0x%016XULL" % sx32_const(imm << 16))
        if op == "ori":              return setr(t, f"{R(s)} | {h(imm)}")
        if op == "andi":             return setr(t, f"{R(s)} & {h(imm)}")
        if op == "xori":             return setr(t, f"{R(s)} ^ {h(imm)}")
        if op == "slti":             return setr(t, f"{RS(s)} < {simm}")
        if op == "sltiu":            return setr(t, f"{R(s)} < (uint64_t)(int64_t){simm}")
        # --- ALU register
        if op in ("addu", "add"):    return setr(d, sx32(f"{RW(s)} + {RW(t)}"))
        if op in ("subu", "sub"):    return setr(d, sx32(f"{RW(s)} - {RW(t)}"))
        if op in ("daddu", "dadd"):  return setr(d, f"{R(s)} + {R(t)}")
        if op in ("dsubu", "dsub"):  return setr(d, f"{R(s)} - {R(t)}")
        if op == "move":             return setr(d, R(s))
        if op == "negu":             return setr(d, sx32(f"0u - {RW(t)}"))
        if op == "and":              return setr(d, f"{R(s)} & {R(t)}")
        if op == "or":               return setr(d, f"{R(s)} | {R(t)}")
        if op == "xor":              return setr(d, f"{R(s)} ^ {R(t)}")
        if op == "nor":              return setr(d, f"~({R(s)} | {R(t)})")
        if op == "not":              return setr(d, f"~{R(s)}")
        if op == "slt":              return setr(d, f"{RS(s)} < {RS(t)}")
        if op == "sltu":             return setr(d, f"{R(s)} < {R(t)}")
        if op == "movz":             return "" if d == 0 else f"if ({R(t)} == 0) {setr(d, R(s))}"
        if op == "movn":             return "" if d == 0 else f"if ({R(t)} != 0) {setr(d, R(s))}"
        # --- shifts
        if op == "sll":              return setr(d, sx32(f"{RW(t)} << {sa}"))
        if op == "srl":              return setr(d, sx32(f"{RW(t)} >> {sa}"))
        if op == "sra":              return setr(d, sx32(f"(uint32_t)({RSW(t)} >> {sa})"))
        if op == "sllv":             return setr(d, sx32(f"{RW(t)} << ({R(s)} & 31)"))
        if op == "srlv":             return setr(d, sx32(f"{RW(t)} >> ({R(s)} & 31)"))
        if op == "srav":             return setr(d, sx32(f"(uint32_t)({RSW(t)} >> ({R(s)} & 31))"))
        if op == "dsll":             return setr(d, f"{R(t)} << {sa}")
        if op == "dsrl":             return setr(d, f"{R(t)} >> {sa}")
        if op == "dsra":             return setr(d, f"(uint64_t)({RS(t)} >> {sa})")
        if op == "dsll32":           return setr(d, f"{R(t)} << {sa + 32}")
        if op == "dsrl32":           return setr(d, f"{R(t)} >> {sa + 32}")
        if op == "dsra32":           return setr(d, f"(uint64_t)({RS(t)} >> {sa + 32})")
        if op == "dsllv":            return setr(d, f"{R(t)} << ({R(s)} & 63)")
        if op == "dsrlv":            return setr(d, f"{R(t)} >> ({R(s)} & 63)")
        if op == "dsrav":            return setr(d, f"(uint64_t)({RS(t)} >> ({R(s)} & 63))")
        # --- multiply / divide (pipeline 0 and 1)
        if op in ("mult", "multu", "mult1", "multu1"):
            p = 1 if op.endswith("1") else 0
            if op.startswith("multu"):
                e = f"(uint64_t){RW(s)} * {RW(t)}"
            else:
                e = f"(uint64_t)((int64_t){RSW(s)} * {RSW(t)})"
            return f"{{ uint64_t _p = {e}; c->lo.ud[{p}] = SX32((uint32_t)_p); c->hi.ud[{p}] = SX32((uint32_t)(_p >> 32)); {setr(d, f'c->lo.ud[{p}]')} }}"
        if op in ("madd", "maddu", "madd1", "maddu1"):
            p = 1 if op.endswith("1") else 0
            if op.startswith("maddu"):
                e = f"(uint64_t){RW(s)} * {RW(t)}"
            else:
                e = f"(uint64_t)((int64_t){RSW(s)} * {RSW(t)})"
            return (f"{{ uint64_t _p = (((uint64_t)(uint32_t)c->hi.ud[{p}] << 32) | (uint32_t)c->lo.ud[{p}]) + {e}; "
                    f"c->lo.ud[{p}] = SX32((uint32_t)_p); c->hi.ud[{p}] = SX32((uint32_t)(_p >> 32)); {setr(d, f'c->lo.ud[{p}]')} }}")
        if op in ("div", "div1"):
            p = 1 if op == "div1" else 0
            return f"rt_div(c, {RSW(s)}, {RSW(t)}, {p});"
        if op in ("divu", "divu1"):
            p = 1 if op == "divu1" else 0
            return f"rt_divu(c, {RW(s)}, {RW(t)}, {p});"
        if op == "mfhi":  return setr(d, "c->hi.ud[0]")
        if op == "mflo":  return setr(d, "c->lo.ud[0]")
        if op == "mfhi1": return setr(d, "c->hi.ud[1]")
        if op == "mflo1": return setr(d, "c->lo.ud[1]")
        if op == "mthi":  return f"c->hi.ud[0] = {R(s)};"
        if op == "mtlo":  return f"c->lo.ud[0] = {R(s)};"
        if op == "mthi1": return f"c->hi.ud[1] = {R(s)};"
        if op == "mtlo1": return f"c->lo.ud[1] = {R(s)};"
        if op == "mfsa":  return setr(d, "c->sa")
        if op == "mtsa":  return f"c->sa = {RW(s)} & 0xF;"
        if op == "mtsab": return f"c->sa = ((({RW(s)}) & 0xF) ^ ({imm} & 0xF));"
        if op == "mtsah": return f"c->sa = (((({RW(s)}) & 0x7) ^ ({imm} & 0x7)) << 1);"
        # --- loads
        if op == "lb":  return setr(t, f"(uint64_t)(int64_t)(int8_t)LB({ea})") or f"(void)LB({ea});"
        if op == "lbu": return setr(t, f"LB({ea})") or f"(void)LB({ea});"
        if op == "lh":  return setr(t, f"(uint64_t)(int64_t)(int16_t)LH({ea})") or f"(void)LH({ea});"
        if op == "lhu": return setr(t, f"LH({ea})") or f"(void)LH({ea});"
        if op == "lw":  return setr(t, sx32(f"LW({ea})")) or f"(void)LW({ea});"
        if op == "lwu": return setr(t, f"LW({ea})") or f"(void)LW({ea});"
        if op == "ld":  return setr(t, f"LD({ea})") or f"(void)LD({ea});"
        if op == "lq":  return f"LQ(&c->r[{t}], {ea});" if t else ";"
        if op in ("lwl", "lwr", "ldl", "ldr"):
            return f"rt_{op}(c, {t}, {ea});"
        # --- stores
        if op == "sb":  return f"SB({ea}, (uint8_t){R(t)});"
        if op == "sh":  return f"SH({ea}, (uint16_t){R(t)});"
        if op == "sw":  return f"SW({ea}, (uint32_t){R(t)});"
        if op == "sd":  return f"SD({ea}, {R(t)});"
        if op == "sq":  return f"SQ({ea}, &c->r[{t}]);"
        if op in ("swl", "swr", "sdl", "sdr"):
            return f"rt_{op}(c, {t}, {ea});"
        # --- FPU
        fs, ft, fd = i.rd, i.rt, i.sa
        if op == "lwc1":  return f"{FU(t)} = LW({ea});"
        if op == "swc1":  return f"SW({ea}, {FU(t)});"
        if op == "mtc1":  return f"{FU(fs)} = {RW(t)};"
        if op == "mfc1":  return setr(t, sx32(FU(fs)))
        if op == "ctc1":  return f"if ({fs} == 31) c->fcr31 = {RW(t)};"
        if op == "cfc1":  return setr(t, sx32(f"({fs} == 31 ? c->fcr31 : 0x2E00)"))
        if op == "add.s":   return f"{F(fd)} = fclamp({F(fs)} + {F(ft)});"
        if op == "sub.s":   return f"{F(fd)} = fclamp({F(fs)} - {F(ft)});"
        if op == "mul.s":   return f"{F(fd)} = fclamp({F(fs)} * {F(ft)});"
        if op == "div.s":   return f"{F(fd)} = fdiv_ps2(c, {F(fs)}, {F(ft)});"
        if op == "sqrt.s":  return f"{F(fd)} = sqrtf(fabsf({F(ft)}));"
        if op == "rsqrt.s": return f"{F(fd)} = fdiv_ps2(c, {F(fs)}, sqrtf(fabsf({F(ft)})));"
        if op == "abs.s":   return f"{F(fd)} = fabsf({F(fs)});"
        if op == "neg.s":   return f"{F(fd)} = -{F(fs)};"
        if op == "mov.s":   return f"{FU(fd)} = {FU(fs)};"
        if op == "max.s":   return f"{F(fd)} = fmaxf({F(fs)}, {F(ft)});"
        if op == "min.s":   return f"{F(fd)} = fminf({F(fs)}, {F(ft)});"
        if op == "adda.s":  return f"c->acc = fclamp({F(fs)} + {F(ft)});"
        if op == "suba.s":  return f"c->acc = fclamp({F(fs)} - {F(ft)});"
        if op == "mula.s":  return f"c->acc = fclamp({F(fs)} * {F(ft)});"
        if op == "madd.s":  return f"{F(fd)} = fclamp(c->acc + {F(fs)} * {F(ft)});"
        if op == "msub.s":  return f"{F(fd)} = fclamp(c->acc - {F(fs)} * {F(ft)});"
        if op == "madda.s": return f"c->acc = fclamp(c->acc + {F(fs)} * {F(ft)});"
        if op == "msuba.s": return f"c->acc = fclamp(c->acc - {F(fs)} * {F(ft)});"
        if op == "cvt.s.w": return f"{F(fd)} = (float){F(fs).replace('.f', '.s')};"
        if op == "cvt.w.s": return f"c->f[{fd}].s = cvt_w_s({F(fs)});"
        if op == "c.f.s":   return "c->fcr31 &= ~FCR_C;"
        if op in ("c.eq.s", "c.lt.s", "c.le.s"):
            cmp = {"c.eq.s": "==", "c.lt.s": "<", "c.le.s": "<="}[op]
            return f"if ({F(fs)} {cmp} {F(ft)}) c->fcr31 |= FCR_C; else c->fcr31 &= ~FCR_C;"
        # --- COP0 / system
        if op == "syscall": return f"rt_syscall(c, {h(i.va)});"
        if op == "break":   return f"rt_break(c, {h(i.va)});"
        if op == "ei":      return "rt_ei(c);"
        if op == "di":      return "rt_di(c);"
        if op == "eret":    return "rt_eret(c); return;"
        if op == "mfc0":    return setr(t, sx32(f"rt_mfc0(c, {i.rd})"))
        if op == "mtc0":    return f"rt_mtc0(c, {i.rd}, {RW(t)});"
        if op == "INVALID":
            w = i.w
            if w >> 26 == 0x10 and (w >> 21) & 31 == 0 and (w >> 11) & 31 == 25:  # mfpc/mfps
                return setr(t, sx32(f"rt_mfperf(c, {w & 0x7FF})"))
            if w >> 26 == 0x10 and (w >> 21) & 31 == 4 and (w >> 11) & 31 == 25:  # mtpc/mtps
                return f"rt_mtperf(c, {w & 0x7FF}, {RW(t)});"
            return f"rt_invalid(c, {h(i.va)}, {h(w)});"
        # --- COP2 / VU0 macro mode
        if op == "lqc2":   return f"LQ(&c->vf[{t}], {ea}); c->vf[0] = VF0;" if t == 0 else f"LQ(&c->vf[{t}], {ea});"
        if op == "sqc2":   return f"SQ({ea}, &c->vf[{t}]);"
        if op in ("qmfc2", "qmfc2.i", "qmfc2.ni"):
            return "" if t == 0 else f"c->r[{t}] = c->vf[{fs}];"
        if op in ("qmtc2", "qmtc2.i", "qmtc2.ni"):
            return ";" if fs == 0 else f"c->vf[{fs}] = c->r[{t}];"
        if op in ("cfc2", "cfc2.i", "cfc2.ni"):
            return setr(t, sx32(f"rt_cfc2(c, {fs})"))
        if op in ("ctc2", "ctc2.i", "ctc2.ni"):
            return f"rt_ctc2(c, {fs}, {RW(t)});"
        if op in ("vcallms",):
            return f"rt_vcallms(c, {((i.w >> 6) & 0x7FFF) * 8});"
        if op in ("vcallmsr",):
            return "rt_vcallms(c, c->vi[27] * 8);"
        if op.startswith("v"):
            v = self.emit_vu(i)
            if v is not None:
                return v
        # --- MMI
        if op[0] == "p" or op in ("qfsrv",):
            return f"mmi_{op.replace('.', '_')}(c, {d}, {s}, {t}, {sa});" + (" c->r[0].ud[0] = c->r[0].ud[1] = 0;" if d == 0 else "")
        self.unhandled[op] += 1
        return f"rt_unimpl(c, {h(i.va)}, \"{op}\");"

    def emit_vu(self, i):
        op, w = i.op, i.w
        dest = (w >> 21) & 15
        comps = [k for k in range(4) if dest & (8 >> k)]
        ft, fs, fd, bc = (w >> 16) & 31, (w >> 11) & 31, (w >> 6) & 31, w & 3

        def VF(n, k): return f"c->vf[{n}].f[{k}]"
        def store(target_acc, n, vals, flags=False):
            out = []
            for k, e in vals:
                out.append(f"_r[{k}] = {e};")
            for k, _ in vals:
                if target_acc:
                    out.append(f"c->vacc.f[{k}] = _r[{k}];")
                elif n != 0:
                    out.append(f"{VF(n, k)} = _r[{k}];")
            if flags:   # FMAC ops update MAC/STATUS (read back with cfc2 $vi17 / $vi16)
                out.append(f"vu0_macro_flags(c, {dest}, _r);")
            return "{ float _r[4] = {0, 0, 0, 0}; " + " ".join(out) + " }"

        m = re.fullmatch(r"v(add|sub|mul|madd|msub|max|mini)(a?)([xyzwqi]?)", op)
        if m:
            kind, acc, bcs = m.groups()
            if kind in ("max", "mini") and acc:
                return None
            def src2(k):
                if bcs in "xyzw" and bcs:
                    return VF(ft, VU_BC.index(bcs))
                if bcs == "q": return "c->vq"
                if bcs == "i": return "c->vi_f[21]"
                return VF(ft, k)
            vals = []
            for k in comps:
                a, b = VF(fs, k), src2(k)
                e = {"add": f"{a} + {b}", "sub": f"{a} - {b}", "mul": f"{a} * {b}",
                     "madd": f"c->vacc.f[{k}] + {a} * {b}", "msub": f"c->vacc.f[{k}] - {a} * {b}",
                     "max": f"fmaxf({a}, {b})", "mini": f"fminf({a}, {b})"}[kind]
                vals.append((k, f"fclamp({e})"))
            return store(bool(acc), fd, vals, flags=kind not in ("max", "mini"))
        if op == "vopmula":
            return ("{ float _r[4] = {fclamp(c->vf[%d].f[1]*c->vf[%d].f[2]), fclamp(c->vf[%d].f[2]*c->vf[%d].f[0]), fclamp(c->vf[%d].f[0]*c->vf[%d].f[1]), 0}; "
                    "c->vacc.f[0]=_r[0]; c->vacc.f[1]=_r[1]; c->vacc.f[2]=_r[2]; vu0_macro_flags(c, 14, _r); }") % (fs, ft, fs, ft, fs, ft)
        if op == "vopmsub":
            wr = "" if fd == 0 else "c->vf[%d].f[0]=_r[0]; c->vf[%d].f[1]=_r[1]; c->vf[%d].f[2]=_r[2]; " % (fd, fd, fd)
            return ("{ float _r[4] = {fclamp(c->vacc.f[0]-c->vf[%d].f[1]*c->vf[%d].f[2]), fclamp(c->vacc.f[1]-c->vf[%d].f[2]*c->vf[%d].f[0]), fclamp(c->vacc.f[2]-c->vf[%d].f[0]*c->vf[%d].f[1]), 0}; "
                    % (fs, ft, fs, ft, fs, ft)) + wr + "vu0_macro_flags(c, 14, _r); }"
        if op in ("vmove", "vabs", "vmr32") or re.fullmatch(r"v(ftoi|itof)(0|4|12|15)", op):
            if ft == 0:
                return ";"
            parts = []
            for k in comps:
                src = f"_s.f[{k}]"
                if op == "vmove":   parts.append(f"c->vf[{ft}].f[{k}] = _s.f[{k}];")
                elif op == "vabs":  parts.append(f"c->vf[{ft}].f[{k}] = fabsf(_s.f[{k}]);")
                elif op == "vmr32": parts.append(f"c->vf[{ft}].f[{k}] = _s.f[{(k + 1) & 3}];")
                else:
                    sh = int(re.search(r"\d+$", op).group())
                    if op.startswith("vftoi"):
                        parts.append(f"c->vf[{ft}].sw[{k}] = vu_ftoi(_s.f[{k}], {sh});")
                    else:
                        parts.append(f"c->vf[{ft}].f[{k}] = (float)_s.sw[{k}] * {1.0 / (1 << sh)!r}f;")
            return f"{{ R128 _s = c->vf[{fs}]; " + " ".join(parts) + " }"
        fsf, ftf = (w >> 21) & 3, (w >> 23) & 3
        if op == "vdiv":   return f"c->vq = fdiv_ps2(c, c->vf[{fs}].f[{fsf}], c->vf[{ft}].f[{ftf}]);"
        if op == "vsqrt":  return f"c->vq = sqrtf(fabsf(c->vf[{ft}].f[{ftf}]));"
        if op == "vrsqrt": return f"c->vq = fdiv_ps2(c, c->vf[{fs}].f[{fsf}], sqrtf(fabsf(c->vf[{ft}].f[{ftf}])));"
        if op == "vsqi":   return f"rt_vsqi(c, {dest}, {fs}, {ft});"
        if op == "vlqi":   return f"rt_vlqi(c, {dest}, {ft}, {fs});"
        return None

    # ------------------------------------------------------------ output
    @staticmethod
    def put(path, text):
        try:
            if open(path).read() == text:
                return False
        except FileNotFoundError:
            pass
        open(path, "w").write(text)
        return True

    def write(self, outdir, nfiles):
        os.makedirs(outdir, exist_ok=True)
        self.entry_set = set(self.entries)
        hdr = ["/* generated by recomp/recomp.py from the user's own SLUS_218.55 - do not commit */",
               "#pragma once", '#include "ps2rt.h"', "#include <setjmp.h>"]
        for e in self.entries:
            hdr.append(f"void {self.fname(e)}(R5900 *c);")
        changed = self.put(os.path.join(outdir, "funcs.h"), "\n".join(hdr) + "\n")

        per = (len(self.starts) + nfiles - 1) // nfiles
        for k in range(nfiles):
            chunk = range(k * per, min(len(self.starts), (k + 1) * per))
            out = ['#include "funcs.h"', "#if defined(_MSC_VER)", "#pragma warning(disable: 4102 4100 4189)", "#endif"]
            for fi in chunk:
                self.emit_func(fi, out)
            changed += self.put(os.path.join(outdir, f"recomp_{k:03d}.c"), "\n\n".join(out) + "\n")

        tab = ['#include "funcs.h"', "const RtFuncEntry rt_func_table[] = {"]
        for e in self.entries:
            tab.append(f"    {{ {h(e)}, {self.fname(e)} }},")
        tab.append("};")
        tab.append(f"const unsigned rt_func_count = {len(self.entries)};")
        changed += self.put(os.path.join(outdir, "table.c"), "\n".join(tab) + "\n")
        print(f"{changed} output files changed; {self.delay_dup} delay slots are also branch targets")
        if self.unhandled:
            print("UNHANDLED opcodes:", dict(self.unhandled))
        else:
            print("all opcodes handled")


def sx32_const(v):
    v &= 0xFFFFFFFF
    return v | 0xFFFFFFFF00000000 if v & 0x80000000 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elf")
    ap.add_argument("out")
    ap.add_argument("--src", nargs="+", default=["src/main_text.c", "src/text2.c"])
    ap.add_argument("--symbols", default="config/symbol_addrs.txt")
    ap.add_argument("--files", type=int, default=256)
    ap.add_argument("--overrides", default="config/hle_overrides.txt")
    a = ap.parse_args()
    r = Recomp(a.elf)
    r.load_functions(a.src, a.symbols)
    r.load_overrides(a.overrides)
    r.analyze()
    r.write(a.out, a.files)


if __name__ == "__main__":
    main()
