---
name: inkverify-project
description: Maintain, run, train, test and deploy the InkVerify handwriting/document screening Flask project. Use whenever the user mentions InkVerify, signature verification with the Random Forest model, train_model.py, the document differencing viewer, CEDAR dataset preparation, or hosting this app on GitHub/Render/Hugging Face, even if they only say "my signature project".
---

# InkVerify project

Flask app: `app.py` routes, `signature_utils.py` (preprocess, 10 pair features, overlay, indicators),
`document_utils.py` (ORB+RANSAC alignment, writes `<prefix>_ref/_qry/_diff.jpg`), `train_model.py`,
`prepare_dataset.py`, `make_demo_data.py`, `templates/`, `static/`, `tests/`.

## Rules that must not be broken
- No model file -> show "MODEL NOT TRAINED", never an invented score.
- Training evaluation is writer-independent (writers never shared between train/val/test).
- Synthetic demo data is for pipeline tests only; `DEMO_DATA.txt` marks it and the UI shows a demo warning. Never present its metrics as real accuracy.
- Changing `pair_features_from` or preprocessing requires bumping `FEATURE_VERSION` so old models are rejected, then retraining.
- Uploads are deleted in `finally`; only `doc_<32hex>_(ref|qry|diff).jpg` files are served from `/uploads/`.
- Results are screening aids, never proof of forgery; keep the disclaimer.

## Common tasks
- Run: `python -m pip install -r requirements.txt`, `python train_model.py`, `python app.py`.
- Real data: `python prepare_dataset.py --source <folder> --clean` then train; copy numbers from `models/metrics.json` into the report.
- Test: `python -m unittest discover -s tests -v`.
- Deploy: follow `DEPLOY.md`; set `INKVERIFY_PASSWORD`; commit the model file only if the user accepts it being public.
- UI changes: CSS variables in `static/style.css` drive light/dark; theme switch is `#theme-switch` in `base.html` handled in `app.js`. Beware `.viewer img` specificity when hiding `.v-diff`.
- Windows pitfalls: if `import cv2` lacks `HOGDescriptor`, uninstall every opencv-* package and install only `opencv-python-headless`; run commands one at a time inside the activated venv.
