"""Train the pair-wise signature consistency model (writer-independent evaluation).

Dataset layout:  data/signatures/<writer_id>/genuine/*.png  and  forged/*.png
Only use samples you have permission to use.

Pair types
  positive            genuine  vs genuine   (same writer)
  negative (skilled)  genuine  vs forged    (same writer)
  negative (random)   genuine  vs genuine   (different writer)

Writers are split into train / validation / test groups, so no writer appears in more
than one group (this measures generalisation to unseen people). The decision threshold
is tuned on the validation writers (equal-error-rate point) and metrics are computed
once on the untouched test writers.

Usage:  python train_model.py [--data data/signatures] [--seed 42]
"""
import argparse, itertools, json, random, time
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from signature_utils import (load_signature, image_features, pair_features_from,
                             FEATURE_VERSION, FEATURE_NAMES)

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
MODEL = Path("models/inkverify_model.joblib")
METRICS = Path("models/metrics.json")
MAX_POS_PER_WRITER = 400
MAX_NEG_PER_WRITER = 400


def list_images(folder):
    return sorted(p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)


def load_dataset(root):
    """-> {writer: {"genuine": [feat...], "forged": [feat...]}}  (features computed once)."""
    data, skipped = {}, 0
    for wdir in sorted(p for p in root.iterdir() if p.is_dir()):
        entry = {}
        for kind in ("genuine", "forged"):
            feats = []
            for path in list_images(wdir / kind):
                try:
                    feats.append(image_features(load_signature(path)))
                except ValueError as exc:
                    skipped += 1
                    print(f"  skipped {path}: {exc}")
            entry[kind] = feats
        if len(entry["genuine"]) >= 2:
            data[wdir.name] = entry
        else:
            print(f"  writer {wdir.name} ignored (needs >= 2 readable genuine samples)")
    if skipped:
        print(f"Skipped {skipped} unreadable image(s).")
    return data


def build_pairs(data, writers, rng):
    X, y, kind = [], [], []     # kind: 1 positive, 0 skilled negative, 2 random negative
    others_pool = {w: [f for o in writers if o != w for f in data[o]["genuine"]] for w in writers}
    for w in writers:
        gen, forg = data[w]["genuine"], data[w]["forged"]
        pos = list(itertools.combinations(range(len(gen)), 2))
        rng.shuffle(pos)
        for i, j in pos[:MAX_POS_PER_WRITER]:
            X.append(pair_features_from(gen[i], gen[j])); y.append(1); kind.append(1)
        skilled = [(a, b) for a in range(len(gen)) for b in range(len(forg))]
        rng.shuffle(skilled)
        n_skilled = min(len(skilled), MAX_NEG_PER_WRITER)
        for a, b in skilled[:n_skilled]:
            X.append(pair_features_from(gen[a], forg[b])); y.append(0); kind.append(0)
        pool = others_pool[w]
        n_rand = min(len(pool), max(len(pos[:MAX_POS_PER_WRITER]) // 2, n_skilled // 2))
        for _ in range(n_rand if pool else 0):
            X.append(pair_features_from(rng.choice(gen), rng.choice(pool))); y.append(0); kind.append(2)
    return np.asarray(X, np.float32), np.asarray(y), np.asarray(kind)


def far_frr(y, prob, t):
    neg, pos = prob[y == 0], prob[y == 1]
    far = float(np.mean(neg >= t)) if len(neg) else float("nan")
    frr = float(np.mean(pos < t)) if len(pos) else float("nan")
    return far, frr


def pick_threshold(y, prob):
    best_t, best_gap = 0.5, 9.0
    for t in np.linspace(0.05, 0.95, 91):
        far, frr = far_frr(y, prob, t)
        if abs(far - frr) < best_gap:
            best_t, best_gap = float(t), abs(far - frr)
    return round(best_t, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/signatures")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(args.data)
    if not root.exists():
        raise SystemExit(f"Dataset folder '{root}' not found. Create data/signatures/<writer>/genuine|forged "
                         "or run:  python make_demo_data.py   (synthetic demo data)")
    t0 = time.time()
    print("Loading images and extracting features (once per image)...")
    data = load_dataset(root)
    writers = sorted(data)
    if len(writers) < 3:
        raise SystemExit("Need at least 3 writers with >= 2 genuine samples for a writer-independent split.")
    rng = random.Random(args.seed)
    rng.shuffle(writers)
    n = len(writers)
    if n >= 6:
        n_test, n_val = max(1, round(n * .15)), max(1, round(n * .15))
        test_w, val_w, train_w = writers[:n_test], writers[n_test:n_test + n_val], writers[n_test + n_val:]
    else:
        test_w, val_w, train_w = writers[:1], [], writers[1:]
        print("WARNING: fewer than 6 writers - no validation group, threshold left at 0.50.")
    print(f"Writers  train={len(train_w)}  val={len(val_w)}  test={len(test_w)}  (no writer overlap)")

    Xtr, ytr, _ = build_pairs(data, train_w, rng)
    Xte, yte, kte = build_pairs(data, test_w, rng)
    if len(set(ytr)) < 2 or len(ytr) < 50 or len(set(yte)) < 2:
        raise SystemExit("Not enough labelled pairs. Add more genuine/forged samples per writer.")
    print(f"Train pairs: {len(ytr)} (genuine-pair share {ytr.mean():.2f}) | Test pairs: {len(yte)}")

    model = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2,
                                   class_weight="balanced_subsample", n_jobs=-1, random_state=args.seed)
    model.fit(Xtr, ytr)

    threshold = 0.5
    if val_w:
        Xva, yva, _ = build_pairs(data, val_w, rng)
        threshold = pick_threshold(yva, model.predict_proba(Xva)[:, 1])
    prob = model.predict_proba(Xte)[:, 1]
    pred = (prob >= threshold).astype(int)

    print(f"\nDecision threshold (validation EER point): {threshold}")
    print(classification_report(yte, pred, target_names=["needs_review", "consistent_pair"], zero_division=0))
    cm = confusion_matrix(yte, pred)
    print("Confusion matrix [rows=true, cols=pred]:\n", cm)
    auc = float(roc_auc_score(yte, prob))
    far, frr = far_frr(yte, prob, threshold)
    far_skilled = far_frr(yte[kte != 2], prob[kte != 2], threshold)[0]
    far_random = far_frr(yte[kte != 0], prob[kte != 0], threshold)[0]
    print(f"ROC-AUC {auc:.4f} | FAR {far:.4f} | FRR {frr:.4f} | FAR(skilled) {far_skilled:.4f} | FAR(random) {far_random:.4f}")

    importances = dict(zip(FEATURE_NAMES, [round(float(v), 4) for v in model.feature_importances_]))
    metrics = {
        "demo_data": (root / "DEMO_DATA.txt").exists(),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "writers": {"train": len(train_w), "val": len(val_w), "test": len(test_w)},
        "pairs": {"train": int(len(ytr)), "test": int(len(yte))},
        "threshold": threshold, "roc_auc": round(auc, 4),
        "far": round(far, 4), "frr": round(frr, 4),
        "far_skilled": round(far_skilled, 4), "far_random": round(far_random, 4),
        "confusion_matrix": cm.tolist(),
        "report": classification_report(yte, pred, target_names=["needs_review", "consistent_pair"],
                                        output_dict=True, zero_division=0),
        "feature_importance": importances,
    }
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "feature_version": FEATURE_VERSION,
                 "feature_names": FEATURE_NAMES, "metrics": metrics}, MODEL, compress=3)
    METRICS.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model   -> {MODEL}\nSaved metrics -> {METRICS}   (use these numbers in report Chapter 6.3)")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
