# InkVerify - Examiner Workbench (Prototype)

InkVerify is a local Flask application with two main tools for document and signature examination.

## Features

### 1. Signature Comparison

- Preprocessing of signature images.
- HOG + shape/profile feature extraction.
- Pair-feature generation.
- Random Forest classification.
- Result page includes:
  - Score gauge.
  - Overlay of the closest known specimen on the questioned signature.
  - Six similarity indicators.
  - Per-specimen comparison scores.

### 2. Document Differencing

- ORB + RANSAC image alignment.
- Illumination normalization.
- Difference mask generation.
- Interactive result viewer with:
  - Swipe comparison.
  - Heatmap.
  - Flicker mode.
  - Detected region list.
  - Saved analysis summary.

## Project Structure

```text
InkVerify/
|-- app.py                  # Routes, password gate, gauge/summary helpers
|-- signature_utils.py      # Preprocessing, features, overlay, indicators
|-- document_utils.py       # Alignment and differencing
|-- train_model.py          # Writer-independent training and metrics.json
|-- prepare_dataset.py      # Dataset conversion and preparation
|-- make_demo_data.py       # Synthetic data for pipeline tests only
|-- templates/              # HTML templates
|-- static/                 # CSS, JavaScript and static assets
|-- tests/                  # Unit tests
|-- Procfile                # Hosting configuration
|-- render.yaml             # Render deployment configuration
|-- Dockerfile              # Docker configuration
|-- .dockerignore
|-- DEPLOY.md               # GitHub and hosting instructions
|-- requirements.txt
|-- README.md
|-- data/signatures/        # Signature dataset (never commit)
|-- models/                 # Trained model and evaluation metrics
|-- uploads/                # Temporary uploaded files
```

## Setup

Create and activate a virtual environment.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Model Training

### Using Real Dataset (Recommended)

Download a dataset you are authorized to use, such as CEDAR.

Prepare the dataset:

```bash
python prepare_dataset.py --source "C:\path\to\CEDAR" --clean
```

Train the model:

```bash
python train_model.py
```

### Using Synthetic Demo Data

For pipeline testing only:

```bash
python make_demo_data.py
python train_model.py
```

**Important:** Synthetic demo data consists of randomly generated curves, not genuine handwriting. Never report its evaluation metrics as real-world signature verification performance.

### Evaluation Metrics

`train_model.py` calculates:

- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)

The evaluation results are saved in:

```text
models/metrics.json
```

These metrics can be used in Chapter 6.3 of the project report.

## Run the Application

Start the Flask application:

```bash
python app.py
```

Open your browser:

```text
http://127.0.0.1:5000
```

### Access from a Phone on the Same Wi-Fi

Set the host environment variable in PowerShell:

```powershell
$env:INKVERIFY_HOST="0.0.0.0"
```

Run the application:

```bash
python app.py
```

Open the following address on your phone:

```text
http://<PC-IP>:5000
```

Replace `<PC-IP>` with your computer's local IP address.

## Environment Variables

InkVerify supports the following environment variables:

| Variable | Purpose |
|---|---|
| `INKVERIFY_PASSWORD` | Optional application password |
| `INKVERIFY_SECRET` | Flask session secret |
| `INKVERIFY_HOST` | Application host address |
| `PORT` | Application port |
| `INKVERIFY_DEBUG` | Enable debug mode (`1`) |

## Important Notes

- Model evaluation is writer-independent.
- The decision threshold is tuned using validation writers.
- Signature uploads are deleted immediately after analysis.
- Comparison images are deleted after one hour.
- Model scores are not calibrated probabilities.
- Results are intended for examination assistance and are not legal proof of forgery.

## Testing

Run the unit tests using:

```bash
python -m unittest discover -s tests -v
```
