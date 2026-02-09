#!/usr/bin/env python3
"""Fix Windows backslash filenames from runpodctl transfer."""
import os
import shutil

def fix_filenames(base_dir="."):
    os.chdir(base_dir)
    os.makedirs("raw_methusdt/cache", exist_ok=True)

    moved = 0
    for f in os.listdir("."):
        if "raw_methusdt" in f and ("\\" in f):
            name = f.split("\\")[-1]
            if "cache" in f:
                dest = "raw_methusdt/cache/" + name
            else:
                dest = "raw_methusdt/" + name
            shutil.move(f, dest)
            print(f"OK: {name}")
            moved += 1

    print(f"\nMoved {moved} files")
    print(f"Data files: {len(os.listdir('raw_methusdt')) - 1}")
    print(f"Cache files: {len(os.listdir('raw_methusdt/cache'))}")

if __name__ == "__main__":
    fix_filenames()
