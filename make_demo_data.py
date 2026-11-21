"""Generate a SYNTHETIC signature-like dataset so the pipeline can be tested end to end.

!! These are random curves, NOT real handwriting. Use it only to check that training
!! and the web app run. Never report its metrics as real signature-verification accuracy.

Usage:  python make_demo_data.py [--writers 12] [--genuine 12] [--forged 8] [--seed 7]
Output: data/signatures/writer_XXX/genuine|forged/*.png
"""
import argparse
from pathlib import Path
import cv2
import numpy as np

W, H = 600, 250


def catmull_rom(pts, n=24):
    p = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-2])
    return np.array(out)


def make_writer_style(rng):
    strokes = []
    for _ in range(rng.integers(2, 5)):
        n = int(rng.integers(5, 9))
        x0 = rng.uniform(40, 200)
        xs = np.sort(rng.uniform(x0, x0 + rng.uniform(150, 360), n))
        ys = rng.uniform(50, 200, n)
        strokes.append(np.stack([xs, ys], 1))
    return strokes


def render(strokes, rng, jitter, rot_deg, scale_range, thick):
    img = np.full((H, W), 255, np.uint8)
    ang = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    s = rng.uniform(*scale_range)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    c = np.array([W / 2, H / 2])
    for pts in strokes:
        q = pts + rng.normal(0, jitter, pts.shape)
        q = ((q - c) @ R.T) * s + c
        curve = catmull_rom(q).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [curve], False, 0, int(rng.integers(*thick)), cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--writers", type=int, default=12)
    ap.add_argument("--genuine", type=int, default=12)
    ap.add_argument("--forged", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/signatures")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = Path(a.out)
    for w in range(1, a.writers + 1):
        style = make_writer_style(rng)
        wd = root / f"writer_{w:03d}"
        (wd / "genuine").mkdir(parents=True, exist_ok=True)
        (wd / "forged").mkdir(parents=True, exist_ok=True)
        for i in range(a.genuine):
            cv2.imwrite(str(wd / "genuine" / f"g{i+1:02d}.png"),
                        render(style, rng, jitter=5, rot_deg=3, scale_range=(0.95, 1.05), thick=(2, 4)))
        for i in range(a.forged):   # imitation: same layout but shakier / different proportions
            cv2.imwrite(str(wd / "forged" / f"f{i+1:02d}.png"),
                        render(style, rng, jitter=16, rot_deg=7, scale_range=(0.85, 1.15), thick=(2, 4)))
    (root / "DEMO_DATA.txt").write_text("Synthetic random curves - not real handwriting. Do not report metrics as real accuracy.\n")
    print(f"Synthetic demo data written to {root.resolve()}  (NOT real handwriting)")


if __name__ == "__main__":
    main()
