"""Convert a downloaded signature dataset into InkVerify's folder layout.

    data/signatures/<writer>/genuine/*.png
    data/signatures/<writer>/forged/*.png

Supported inputs (auto-detected)
  A) CEDAR-style file names ........ original_<writer>_<n>.png   forgeries_<writer>_<n>.png
  B) Folder style .................. <writer>/<genuine|original|org>/*.png  and  <writer>/<forged|forgery|forg>/*.png
     (also works when the writer folder sits deeper, e.g. dataset/train/<writer>/genuine/...)

Usage
  python prepare_dataset.py --source "C:\\path\\to\\CEDAR" [--out data/signatures] [--clean] [--max-writers 60]

--clean removes old writer_* folders (e.g. the synthetic demo data) and DEMO_DATA.txt first.
Check the dataset's licence and only use data you may use.
"""
import argparse, re, shutil, sys
from pathlib import Path

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NAME = re.compile(r"(original|orig|genuine|org|forgeries|forgery|forged|forg)[_\- ]*(\d+)[_\- ]*(\d+)", re.I)
GEN_WORDS = ("genuine", "original", "orig", "org", "real")
FOR_WORDS = ("forged", "forgery", "forgeries", "forg", "fake")


def kind_of(word):
    w = word.lower()
    return "forged" if w.startswith("forg") else "genuine"


def scan(source):
    found = {}                                   # writer -> {"genuine": [...], "forged": [...]}
    images = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXT]
    if not images:
        sys.exit(f"No images found under {source}")
    # style A: file names carry writer + kind
    for p in images:
        m = NAME.search(p.stem)
        if m:
            found.setdefault(f"writer_{int(m.group(2)):03d}", {"genuine": [], "forged": []})[kind_of(m.group(1))].append(p)
    if found:
        return found
    # style B: .../<writer>/<kind>/<image>
    for p in images:
        kind_dir = p.parent.name.lower()
        if any(k in kind_dir for k in FOR_WORDS):
            kind = "forged"
        elif any(k in kind_dir for k in GEN_WORDS):
            kind = "genuine"
        else:
            continue
        writer = re.sub(r"\W+", "_", p.parent.parent.name) or "writer"
        found.setdefault(f"writer_{writer}", {"genuine": [], "forged": []})[kind].append(p)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="data/signatures")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--max-writers", type=int, default=0)
    a = ap.parse_args()
    src, out = Path(a.source), Path(a.out)
    if not src.exists():
        sys.exit(f"Source folder not found: {src}")
    found = scan(src)
    if not found:
        sys.exit("Could not recognise the layout. See the notes at the top of this file.")
    writers = sorted(found)
    if a.max_writers:
        writers = writers[:a.max_writers]
    if a.clean and out.exists():
        for d in out.glob("writer_*"):
            shutil.rmtree(d, ignore_errors=True)
        (out / "DEMO_DATA.txt").unlink(missing_ok=True)
    total_g = total_f = 0
    for w in writers:
        for kind in ("genuine", "forged"):
            dest = out / w / kind
            dest.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(sorted(found[w][kind]), 1):
                shutil.copy2(p, dest / f"{kind[0]}{i:02d}{p.suffix.lower()}")
            n = len(found[w][kind])
            total_g, total_f = total_g + (n if kind == "genuine" else 0), total_f + (n if kind == "forged" else 0)
    print(f"Prepared {len(writers)} writers -> {out.resolve()}  ({total_g} genuine, {total_f} forged images)")
    print("Next:  python train_model.py")


if __name__ == "__main__":
    main()
