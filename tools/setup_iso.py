#!/usr/bin/env python3
"""Find the user's ISO in game_iso/, verify it, and extract it to iso/.

This project ships NO game content. Everything under iso/, asm/, assets/ and
build/ is derived from the ISO you provide, and must never be committed.

Usage: python tools/setup_iso.py [--iso PATH] [--force]
"""
import argparse, glob, hashlib, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_iso import extract  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_ISO_DIR = os.path.join(ROOT, "game_iso")
OUT = os.path.join(ROOT, "iso")

# NPPL Championship Paintball 2009 (USA), SLUS-21855
KNOWN_ISO_SHA1 = {
    "8db77412f8d920d6b742269deb8c5c0ec7b5e4f1": "NPPL Championship Paintball 2009 (USA)",
}
VOLUME_ID = "NPPL_PAINTBALL"
REQUIRED = {  # relative path -> sha1
    "SLUS_218.55": "3a54509fd715deb3080e05c32a009ba35e1f4568",
    "DATA/DATA.ARC": "a8525b2048dfd4364086dca051bd6be9117b2df7",
}

def sha1_file(path, chunk=1 << 24):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)

def volume_id(path):
    with open(path, "rb") as f:
        f.seek(16 * 2048)
        pvd = f.read(2048)
    if pvd[1:6] != b"CD001":
        return None
    return pvd[40:72].decode("ascii", "replace").strip()

def find_iso():
    isos = [p for p in glob.glob(os.path.join(GAME_ISO_DIR, "*")) if p.lower().endswith(".iso")]
    if not isos:
        sys.exit(
            "No ISO found.\n"
            f"Put your own dump of NPPL Championship Paintball 2009 (USA) (SLUS-21855) in:\n  {GAME_ISO_DIR}\n"
            "Any filename ending in .iso works. BIN/CUE, CHD and CSO must be converted to ISO first."
        )
    if len(isos) > 1:
        sys.exit("More than one .iso in game_iso/; keep only one:\n  " + "\n  ".join(isos))
    return isos[0]

def verify_extracted():
    ok = True
    for rel, want in REQUIRED.items():
        p = os.path.join(OUT, rel)
        if not os.path.exists(p):
            print(f"  MISSING {rel}"); ok = False; continue
        got = sha1_file(p)
        if got != want:
            print(f"  BAD     {rel} (sha1 {got}, expected {want})"); ok = False
        else:
            print(f"  ok      {rel}")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso", help="path to ISO (default: the single .iso in game_iso/)")
    ap.add_argument("--force", action="store_true", help="re-extract even if iso/ looks valid")
    a = ap.parse_args()

    if not a.force and os.path.exists(os.path.join(OUT, "SLUS_218.55")):
        print("iso/ already extracted; verifying...")
        if verify_extracted():
            print("Setup OK.")
            return
        print("Extracted files are wrong; re-extracting.")

    iso = a.iso or find_iso()
    print(f"Using {iso}")
    vid = volume_id(iso)
    if vid is None:
        sys.exit("That file isn't an ISO9660 image (it may be BIN/CUE, CHD or CSO). Convert it to ISO first.")
    if vid != VOLUME_ID:
        sys.exit(f"Wrong disc: volume ID is '{vid}', expected '{VOLUME_ID}'. Only the USA release (SLUS-21855) is supported.")

    print("Hashing ISO...")
    h = sha1_file(iso)
    if h in KNOWN_ISO_SHA1:
        print(f"  verified: {KNOWN_ISO_SHA1[h]}")
    else:
        print(f"  unrecognized ISO sha1 {h}; continuing, then checking the game files individually")

    print("Extracting to iso/ (skipping TRASH/ padding)...")
    extract(iso, OUT, quiet=True)
    if not verify_extracted():
        sys.exit("The game files don't match the supported version. Only the USA release (SLUS-21855) is supported.")
    print("Setup OK.")

if __name__ == "__main__":
    main()
