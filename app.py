"""InkVerify - local Flask app for signature comparison and document differencing.

Run:   python app.py            -> http://127.0.0.1:5000
Host:  gunicorn app:app         (Render, Railway, Docker ...)

Environment variables (all optional)
  INKVERIFY_SECRET     session key
  INKVERIFY_PASSWORD   if set, the whole site asks for this password (user name: anything)
  INKVERIFY_HOST       bind address for `python app.py` (default 127.0.0.1, use 0.0.0.0 for phone testing)
  PORT                 port (default 5000)
  INKVERIFY_DEBUG=1    Flask debug mode
"""
import math
import os
import re
import secrets
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
from flask import (Flask, Response, abort, flash, redirect, render_template, request,
                   send_from_directory, url_for)

from document_utils import compare_documents
from signature_utils import (FEATURE_VERSION, image_features, indicators, load_signature,
                             overlay_png_b64, pair_features_from)

VERSION = "1.2"
BASE = Path(__file__).resolve().parent
UPLOAD = BASE / "uploads"
UPLOAD.mkdir(exist_ok=True)
MODEL_PATH = BASE / "models" / "inkverify_model.joblib"
DEFAULT_THRESHOLD = 0.60            # used only if the model file stores no tuned threshold
ALLOWED = {"png", "jpg", "jpeg", "bmp"}
MAX_REFERENCES = 10
DOC_FILE = re.compile(r"^doc_[0-9a-f]{32}_(ref|qry|diff)\.jpg$")
PASSWORD = os.environ.get("INKVERIFY_PASSWORD")

app = Flask(__name__)
app.secret_key = os.environ.get("INKVERIFY_SECRET") or secrets.token_hex(16)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024      # phone photos are large


# ---------- helpers ----------
def cleanup_old_uploads(max_age_seconds=3600):
    now = time.time()
    for f in UPLOAD.iterdir():
        if f.is_file() and f.name != ".gitkeep" and now - f.stat().st_mtime > max_age_seconds:
            f.unlink(missing_ok=True)


def save_upload(file):
    name = file.filename or ""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED:
        raise ValueError("Use PNG, JPG or BMP image files only.")
    path = UPLOAD / f"{uuid4().hex}.{ext}"      # server-generated name, user name never used
    file.save(path)
    return path


@lru_cache(maxsize=1)
def _load_model(mtime):
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError("The saved model is outdated. Run:  python train_model.py")
    return bundle


def get_model():
    """Return the trained model bundle or None. Reloads automatically after retraining."""
    if not MODEL_PATH.exists():
        return None
    return _load_model(MODEL_PATH.stat().st_mtime)


def _prob(bundle, fa, fb):
    return float(bundle["model"].predict_proba(pair_features_from(fa, fb).reshape(1, -1))[0, 1])


def _case_id():
    return "IV-" + uuid4().hex[:6].upper()


def _stamp():
    return datetime.now().strftime("%d %b %Y, %H:%M")


def gauge_geometry(score, threshold):
    """Coordinates for the semicircle gauge drawn in result.html (viewBox 0 0 120 70)."""
    ang = math.pi * (1 - threshold)
    cx, cy, r = 60, 62, 50
    return {
        "x1": round(cx + (r - 8) * math.cos(ang), 2), "y1": round(cy - (r - 8) * math.sin(ang), 2),
        "x2": round(cx + (r + 7) * math.cos(ang), 2), "y2": round(cy - (r + 7) * math.sin(ang), 2),
        "fill": round(max(0.0, min(1.0, score)) * 100, 1),
    }


@app.context_processor
def inject_globals():
    info = {"state": "missing"}
    try:
        b = get_model()
        if b:
            m = b.get("metrics") or {}
            w = m.get("writers") or {}
            info = {"state": "ready", "threshold": b.get("threshold", DEFAULT_THRESHOLD),
                    "writers": sum(w.values()) if w else None, "demo": bool(m.get("demo_data")),
                    "auc": m.get("roc_auc")}
    except ValueError:
        info = {"state": "outdated"}
    return {"model_info": info, "version": VERSION}


@app.before_request
def password_gate():
    if not PASSWORD:
        return None
    auth = request.authorization
    if not auth or not secrets.compare_digest((auth.password or "").encode(), PASSWORD.encode()):
        return Response("Password required.", 401, {"WWW-Authenticate": 'Basic realm="InkVerify"'})
    return None


@app.after_request
def headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.errorhandler(413)
def too_large(_):
    flash("Upload too large (limit 40 MB in total). Try smaller photos.")
    return redirect(url_for("index"))


def signature_summary(case, status, score, threshold, per_ref, ref_consistency, inds, demo):
    lines = ["INKVERIFY - SIGNATURE COMPARISON", f"Case ref: {case}", f"Generated: {_stamp()}", "",
             f"Result: {status}", f"Mean pair score: {score*100:.1f}%",
             f"Decision threshold: {threshold*100:.0f}%",
             f"Known specimens: {len(per_ref)}", f"Specimen-to-specimen consistency: {ref_consistency*100:.0f}%", ""]
    lines += [f"  K{i}: {v*100:.1f}%" for i, v in enumerate(per_ref, 1)]
    lines += ["", "Similarity indicators:"] + [f"  {n}: {v*100:.0f}%" for n, v in inds]
    if demo:
        lines += ["", "NOTE: model trained on synthetic demo data - software testing only."]
    lines += ["", "Automated screening only. Not proof of forgery. Qualified human review required."]
    return "\n".join(lines)


def document_summary(case, r):
    lines = ["INKVERIFY - DOCUMENT DIFFERENCING", f"Case ref: {case}", f"Generated: {_stamp()}", "",
             f"Candidate changed regions: {r['region_count']}", f"Changed area: {r['changed_percent']}%",
             f"Alignment: {r['alignment']} ({r['inliers']} keypoint matches)"]
    if r["warning"]:
        lines.append(f"Warning: {r['warning']}")
    lines += ["", "Regions (x, y, width, height in px):"]
    lines += [f"  {i}. {x}, {y}, {w}, {h}" for i, (x, y, w, h) in enumerate(r["regions"], 1)] or ["  none"]
    lines += ["", "Highlighted areas may also come from lighting, compression or scan noise.",
              "Automated screening only. Qualified human review required."]
    return "\n".join(lines)


# ---------- routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/verify", methods=["POST"])
def verify():
    cleanup_old_uploads()
    paths = []
    try:
        refs = [f for f in request.files.getlist("references") if f and f.filename]
        query = request.files.get("questioned")
        if len(refs) < 2 or not query or not query.filename:
            raise ValueError("Add at least two known specimens and one questioned signature.")
        if len(refs) > MAX_REFERENCES:
            raise ValueError(f"Use at most {MAX_REFERENCES} known specimens.")
        paths = [save_upload(f) for f in refs]
        q_path = save_upload(query)
        paths.append(q_path)

        bundle = get_model()
        ref_imgs = [load_signature(p) for p in paths[:-1]]          # raises on bad/blank image
        q_img = load_signature(q_path)
        ref_feats = [image_features(i) for i in ref_imgs]
        q_feat = image_features(q_img)
        case = _case_id()
        if bundle is None:
            return render_template("result.html", model_ready=False, case=case,
                                   status="MODEL NOT TRAINED")

        threshold = float(bundle.get("threshold", DEFAULT_THRESHOLD))
        scores = [_prob(bundle, rf, q_feat) for rf in ref_feats]
        score = float(np.mean(scores))
        ref_pairs = [_prob(bundle, ref_feats[i], ref_feats[j])
                     for i in range(len(ref_feats)) for j in range(i + 1, len(ref_feats))]
        ref_consistency = float(np.mean(ref_pairs))
        consistent = score >= threshold
        status = "Consistent with known specimens" if consistent else "Low consistency - review manually"
        best = int(np.argmax(scores))
        inds = indicators(ref_feats, q_feat)
        demo = bool((bundle.get("metrics") or {}).get("demo_data"))
        return render_template(
            "result.html", model_ready=True, case=case, score=score, status=status,
            consistent=consistent, threshold=threshold, per_ref=scores, best=best,
            borderline=abs(score - threshold) < 0.10, ref_consistency=ref_consistency,
            ref_warning=ref_consistency < threshold, metrics=bundle.get("metrics") or {},
            gauge=gauge_geometry(score, threshold), inds=inds,
            overlay=overlay_png_b64(ref_imgs[best], q_img),
            summary=signature_summary(case, status, score, threshold, scores, ref_consistency, inds, demo),
        )
    except Exception as exc:                       # show a clean message, never a stack trace
        flash(str(exc))
        return redirect(url_for("index"))
    finally:
        for p in paths:
            p.unlink(missing_ok=True)


@app.route("/compare", methods=["POST"])
def compare():
    cleanup_old_uploads()
    paths = []
    try:
        original, questioned = request.files.get("original"), request.files.get("questioned_doc")
        if not original or not original.filename or not questioned or not questioned.filename:
            raise ValueError("Add both the original and the questioned scan.")
        a, b = save_upload(original), save_upload(questioned)
        paths = [a, b]
        token = uuid4().hex
        result = compare_documents(a, b, UPLOAD / f"doc_{token}")
        urls = {k: url_for("uploaded", name=Path(v).name) for k, v in result["files"].items()}
        case = _case_id()
        return render_template("document_result.html", result=result, urls=urls, case=case,
                               summary=document_summary(case, result))
    except Exception as exc:
        flash(str(exc))
        return redirect(url_for("index"))
    finally:
        for p in paths:
            p.unlink(missing_ok=True)


@app.route("/uploads/<name>")
def uploaded(name):
    if not DOC_FILE.match(name):           # only generated comparison images are ever served
        abort(404)
    return send_from_directory(UPLOAD, name)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        debug=False
    )