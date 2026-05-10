# Sleep Stage Classification — CI/CD Pipeline

Automated CI/CD for the CNN + Transformer sleep staging model (AI341).

## Pipeline Overview

```
push/PR
  │
  ├─► lint          Flake8, black, isort
  ├─► test          Pytest unit + integration tests
  ├─► validate-data Check preprocessing outputs for NaNs, shape, label range
  │
  ├─► train*        Download PhysioNet data → preprocess → train → save artifact
  ├─► evaluate*     Load results.json → gate on kappa ≥ 0.70 & accuracy ≥ 0.80
  ├─► build*        Build Docker image → push to GHCR
  └─► deploy*       Deploy to staging server
                    (* main branch / manual trigger only)
```

## Quick Start

### 1. Clone and install
```bash
git clone https://github.com/your-org/sleep-staging.git
cd sleep-staging
pip install -r requirements.txt
```

### 2. Run tests locally
```bash
pytest tests/ -v
```

### 3. Validate data pipeline (no real EEG files needed)
```bash
python scripts/validate_data.py --dry-run
```

### 4. Trigger full training manually
Go to **Actions → Sleep Stage Classification CI/CD → Run workflow** and set `run_training = true`.

---

## Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `PHYSIONET_USER` | PhysioNet username (for dataset download) |
| `PHYSIONET_PASS` | PhysioNet password |
| `WANDB_API_KEY` | Weights & Biases API key (optional) |
| `DEPLOY_KEY` | SSH key for staging deployment |

Set these at **Settings → Secrets and variables → Actions**.

---

## Quality Gates

Training is rejected and the pipeline fails if:
- **Accuracy < 0.80**
- **Cohen's Kappa < 0.70**

Your current best model (Epoch 9): Accuracy = 86.4%, Kappa = 0.7373 ✅

---

## Project Structure

```
.
├── .github/workflows/ci-cd.yml   # Main pipeline
├── configs/default.yaml          # Hyperparameters
├── scripts/
│   ├── validate_data.py          # Data quality checks
│   ├── evaluate.py               # Metric gate
│   ├── download_data.py          # PhysioNet downloader
│   ├── preprocess.py             # Preprocessing pipeline
│   └── train.py                  # Training entry point
├── src/                          # Model source code
├── tests/
│   └── unit/
│       └── test_preprocessing.py
├── Dockerfile
└── requirements.txt
```
