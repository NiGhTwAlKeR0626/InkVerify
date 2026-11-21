# InkVerify - examiner workbench (prototype)

Local Flask app with two tools:

1. **Signature comparison** - preprocessing -> HOG + shape/profile features -> pair features -> Random Forest.
   Result page: score gauge, overlay of the closest known specimen on the questioned signature,
   six similarity indicators, per-specimen scores.
2. **Document differencing** - ORB + RANSAC alignment -> illumination normalisation -> difference mask.
   Result page: interactive viewer (swipe, heatmap, flicker), region list, saved summary.

## Structure
```
InkVerify/
|-- app.py                 routes, password gate, gauge/summary helpers
|-- signature_utils.py     preprocessing, features, overlay, indicators
|-- document_utils.py      alignment + differencing (writes _ref/_qry/_diff images)
|-- train_model.py         writer-independent training + metrics.json
|-- prepare_dataset.py     convert CEDAR-style / folder datasets to data/signatures
|-- make_demo_data.py      SYNTHETIC data for pipeline tests only
|-- templates/  static/    UI (light/dark switch, tabs, drag-drop, viewer)
|-- tests/                 python -m unittest discover -s tests -v
|-- Procfile render.yaml Dockerfile .dockerignore   hosting
|-- DEPLOY.md              GitHub + hosting steps
|-- data/signatures/       your dataset (never commit it)
|-- models/                inkverify_model.joblib, metrics.json
`-- uploads/               temporary files (auto-deleted)
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell;  Mac/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train
Real data (recommended): download a dataset you are allowed to use, e.g. CEDAR, then
```bash
python prepare_dataset.py --source "C:\path\to\CEDAR" --clean
python train_model.py
```
Pipeline test only (random curves, not handwriting - never quote these numbers):
```bash
python make_demo_data.py
python train_model.py
```
`train_model.py` prints precision, recall, F1, confusion matrix, ROC-AUC, FAR/FRR and writes
`models/metrics.json` for report Chapter 6.3.

## Run
```bash
python app.py                        # http://127.0.0.1:5000
```
Phone on the same Wi-Fi: `set INKVERIFY_HOST=0.0.0.0` (PowerShell: `$env:INKVERIFY_HOST="0.0.0.0"`), run, open `http://<PC-IP>:5000`.

Options (environment variables): `INKVERIFY_PASSWORD` (lock the site), `INKVERIFY_SECRET`, `INKVERIFY_HOST`, `PORT`, `INKVERIFY_DEBUG=1`.

## Notes
- Evaluation is writer-independent; the threshold is tuned on validation writers.
- Signature uploads are deleted right after analysis; comparison images after 1 hour.
- Scores are not calibrated probabilities and are not legal proof of forgery.
=======
# InkVerify
>>>>>>> 7acf4ebdb79a07e931f574452736f0ab9dc4da8a
