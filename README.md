# 🧠 Deep Learning for Automated Sleep Stage Classification

> **Course:** AI341 - Deep Learning  
> **Instructor:** Talha Ishfaq  
> **Date:** March 2026

**Team Members:**
- Rumaisa Siddiqa — 2023615
- Umme Habiba Malik — 2023737

---

## 📌 What Is This Project?

Sleep is essential for human health. Doctors diagnose sleep disorders (like sleep apnea, insomnia, and narcolepsy) by analyzing a patient's **sleep stages** throughout the night. Traditionally, this is done by a sleep specialist who manually reads hours of brain wave recordings (EEG) and labels each 30-second segment as one of 5 stages:

| Stage | Name | What It Means |
|-------|------|---------------|
| W | Wake | Person is awake |
| N1 | Light Sleep | Transition into sleep |
| N2 | Intermediate Sleep | True sleep begins |
| N3 | Deep Sleep | Most restorative stage |
| REM | Rapid Eye Movement | Dreaming stage |

This manual process takes **2–4 hours per patient** and requires a trained specialist — a resource that is scarce in countries like Pakistan.

**Our project automates this entire process using deep learning.** We built a model that reads raw EEG signals and classifies every 30-second segment into the correct sleep stage — in seconds, not hours.

---

## 🎯 Why We Built This

1. **Clinical need** — Sleep disorder diagnosis is slow, expensive, and specialist-dependent
2. **Pakistan context** — Sleep medicine specialists are extremely limited in Pakistan; an automated tool can help clinics without specialists
3. **Research benchmark** — The PhysioNet Sleep-EDF database is a well-known benchmark, allowing us to compare directly against published research
4. **Academic goal** — Demonstrate a full deep learning pipeline from raw signal to prediction, meeting the AI341 project requirements

---

## 🏗️ Architecture Overview

We use a **two-level CNN + Transformer** architecture:

```
Raw EEG Signal (30 seconds)
        │
        ▼
┌─────────────────────┐
│   Epoch Encoder     │  ← Level 1: CNN
│   (Multi-Scale CNN) │    Extracts features from each 30-sec epoch
│   + Attention Pool  │    Uses kernel sizes 50, 25, 10 to capture
└─────────────────────┘    different time-scale patterns
        │
        ▼ (embedding per epoch)
┌─────────────────────┐
│ Positional Encoding │  ← Tells the model the order of epochs
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│    Transformer      │  ← Level 2: Transformer
│  (3 layers, 4 heads)│    Looks at 20 epochs at once (10 minutes)
│                     │    Understands context: what stage came before
└─────────────────────┘    and after helps classify current stage
        │
        ▼
┌─────────────────────┐
│  Classification     │  ← Predicts sleep stage for each epoch
│  Head (5 classes)   │
└─────────────────────┘
```

### Why CNN + Transformer?

- **CNN alone** looks at each 30-second epoch in isolation — misses context
- **Transformer alone** cannot handle raw 3000-sample time series efficiently
- **Combined**: CNN compresses each epoch into a compact embedding, then Transformer understands the sequence of epochs — mimicking how a human expert reads a full night's hypnogram

---

## 📂 Project Structure

```
sleep-stage-classification/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml              ← Automated CI/CD pipeline (GitHub Actions)
│
├── configs/
│   └── default.yaml               ← All hyperparameters in one place
│
├── scripts/
│   ├── validate_data.py           ← Checks data quality before training
│   └── evaluate.py                ← Gates model on quality thresholds
│
├── tests/
│   └── unit/
│       └── test_preprocessing.py  ← Automated unit tests
│
├── notebooks/
│   └── Sleep_Stage_Classification.ipynb  ← Main project notebook
│
├── models/
│   ├── best_model.pth             ← Trained model weights (11 MB)
│   └── training_history.pkl       ← Loss/accuracy curves data
│
├── reports/
│   ├── training_curves.png        ← Loss and accuracy over epochs
│   ├── confusion_matrix.png       ← Per-class prediction heatmap
│   ├── per_class_f1.png           ← F1 score for each sleep stage
│   ├── hypnograms.png             ← Predicted vs ground truth overnight
│   └── baseline_comparison.png   ← Our model vs published baselines
│
├── results.json                   ← All final metrics saved
├── Dockerfile                     ← Container for deployment
├── requirements.txt               ← All Python dependencies
├── .gitignore                     ← Excludes large data files from Git
└── README.md                      ← This file
```

---

## 🔬 Dataset

**PhysioNet Sleep-EDF Expanded Database**
- Source: https://physionet.org/content/sleep-edfx/1.0.0/
- Format: European Data Format (EDF) polysomnography recordings
- Subjects used: 5 (Night 1 recordings)
- EEG channels: `EEG Fpz-Cz` and `EEG Pz-Oz`
- Sampling rate: 100 Hz
- Epoch length: 30 seconds → 3000 samples per epoch

**Why PhysioNet Sleep-EDF?**
It is the most widely used public sleep dataset, allowing direct comparison with published models like DeepSleepNet and SleepTransformer.

---

## ⚙️ Preprocessing Pipeline

Every EEG recording goes through this pipeline before training:

| Step | What We Do | Why |
|------|-----------|-----|
| Channel selection | Pick `EEG Fpz-Cz`, `EEG Pz-Oz` | Standard clinical channels for sleep |
| Resampling | Resample to 100 Hz if needed | Consistent input size across subjects |
| Bandpass filter | 0.5–35 Hz | Remove noise below 0.5 Hz (drift) and above 35 Hz (muscle artifacts) |
| Annotation mapping | Map stage labels to integers 0–4 | Stage 4 merged into N3 per AASM 2012 standard |
| Invalid epoch removal | Remove `Movement time` and `?` epochs | These cannot be scored reliably |
| Z-score normalization | Per recording, per channel | Removes amplitude differences between subjects and nights |
| Epoch segmentation | 30-second windows | Matches clinical standard for sleep scoring |
| Sequence creation | Groups of 20 epochs (10 minutes) | Gives model temporal context |

---

## 🤖 Model Details

### Epoch Encoder (CNN)
- **3 Multi-Scale CNN blocks**, each using kernel sizes `[50, 25, 10]` → `[25, 10, 5]` → `[10, 5, 3]`
- Each block has: Conv1D → BatchNorm → ReLU → MaxPool → Dropout
- **Attention pooling** at the end: learns which time steps in the epoch matter most
- Output: 128-dimensional embedding per epoch

### Sequence Model (Transformer)
- **3 Transformer encoder layers**, 4 attention heads
- Feed-forward dimension: 256
- Sinusoidal positional encoding (tells the model epoch order)
- Processes 20 consecutive epochs at once (10 minutes of context)

### Classification Head
- Linear(128 → 64) → ReLU → Dropout(0.3) → Linear(64 → 5)
- Outputs one prediction per epoch in the sequence

### Training Setup

| Component | Choice | Reason |
|-----------|--------|--------|
| Loss | Weighted CrossEntropy | Handles class imbalance (N2 dominates ~45%) |
| Optimizer | AdamW | Better weight decay than Adam |
| Scheduler | CosineAnnealingWarmRestarts | Avoids local minima, T₀=10 |
| Batch size | 32 sequences | Memory efficient |
| Max epochs | 50 | With early stopping |
| Early stopping | Patience = 10 | Stops if kappa doesn't improve |
| Augmentation | Gaussian noise + amplitude scaling | Improves generalization |
| Random seed | 42 | Reproducibility |

---

## 📊 Results

| Metric | Our Model | Target |
|--------|-----------|--------|
| Test Accuracy | **86.4%** | ≥ 80% ✅ |
| Cohen's Kappa | **0.7373** | ≥ 0.75 ⚠️ (very close) |

### Comparison with Published Baselines

| Model | Accuracy | Kappa |
|-------|----------|-------|
| Per-epoch CNN (No Context) | 75% | 0.68 |
| DeepSleepNet (CNN + BiLSTM) | 82% | 0.76 |
| SleepTransformer (CNN + Transformer) | 86% | 0.81 |
| **Our Model (CNN + Transformer)** | **86.4%** | **0.737** |

Our model **matches the SleepTransformer accuracy** using only 5 subjects — a much smaller training set than the published papers used.

### Why Cohen's Kappa?
Accuracy alone is misleading when classes are imbalanced. Cohen's Kappa measures **agreement beyond chance** — a kappa of 0.737 means our model is substantially better than random guessing even accounting for the class distribution. Values above 0.60 are considered "substantial agreement" in clinical contexts.

---

## 🔄 CI/CD Pipeline

Every time code is pushed to GitHub, the pipeline runs automatically:

```
git push
    │
    ├─► Lint (flake8, black, isort)
    ├─► Unit Tests (pytest)
    ├─► Data Validation (shape, NaN, label checks)
    │
    └─► [main branch only]
        ├─► Train Model
        ├─► Evaluate (gate: kappa ≥ 0.70, accuracy ≥ 0.80)
        ├─► Build Docker Image
        └─► Deploy to Staging
```

**Quality gates**: if the trained model scores below the thresholds, the pipeline fails and the bad model is never deployed.

---

## 🚀 How to Run

### Prerequisites
- Python 3.12
- GPU recommended (runs on CPU too, just slower)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run in Google Colab (recommended)
1. Open `notebooks/Sleep_Stage_Classification.ipynb` in Google Colab
2. Set runtime to GPU: `Runtime → Change runtime type → T4 GPU`
3. Run all cells from top to bottom

### Run tests locally
```bash
pytest tests/ -v
```

### Validate data pipeline (no EEG files needed)
```bash
python scripts/validate_data.py --dry-run
```

---

## 🐳 Docker

Build and run the inference container:
```bash
docker build -t sleep-classifier .
docker run -p 8000:8000 sleep-classifier
```

---

## 📈 Key Visualizations

| File | What It Shows |
|------|--------------|
| `training_curves.png` | Loss and accuracy improving over 50 epochs |
| `confusion_matrix.png` | Which stages get confused with which |
| `per_class_f1.png` | N1 is hardest (transitional stage), N2 and N3 are easiest |
| `hypnograms.png` | Full overnight prediction vs ground truth for 3 subjects |
| `baseline_comparison.png` | Our model next to DeepSleepNet and SleepTransformer |

---

## ⚠️ Known Limitations

1. **N1 classification is weak** — N1 is a transitional stage that is difficult even for human experts to score consistently
2. **Only 5 subjects** — published models use 40–197 subjects; more data would improve kappa
3. **Class imbalance** — N2 makes up ~45% of all epochs; despite weighted loss, minority stages are harder
4. **Single night** — only Night 1 recordings used; performance may differ on Night 2

---

## 🔮 Future Work

1. Train on more subjects (full 197-recording Sleep-EDF database)
2. Add EOG and EMG channels for better REM detection
3. Implement uncertainty quantification for ambiguous epochs
4. Transfer learning: fine-tune on Pakistani clinical data
5. Real-time streaming inference for bedside monitoring

---

## 📚 References

1. Supratak, A., et al. (2017). **DeepSleepNet**: A model for automatic sleep stage scoring based on raw single-channel EEG. *IEEE TNSRE*.
2. Phan, H., et al. (2022). **SleepTransformer**: Automatic Sleep Staging with Interpretability and Uncertainty Quantification. *IEEE TBME*.
3. Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and **PhysioNet**. *Circulation*. https://physionet.org/content/sleep-edfx/1.0.0/
4. Berry, R. B., et al. (2012). **AASM Scoring Manual** for the Scoring of Sleep and Associated Events. American Academy of Sleep Medicine.

---

## 📄 License

This project is submitted for academic evaluation as part of AI341 at FAST-NUCES. Dataset is publicly available under the PhysioNet Credentialed Health Data License.
