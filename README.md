<div align="center">

# Veridica

**Dual-language fake news detection · English & Arabic**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat)](LICENSE)

Nine model architectures compared across two languages —  
from TF-IDF baselines to fine-tuned DistilBERT and AraBERTv2.

[**Live Demo**](https://share.streamlit.io) · [**Notebook**](notebooks/nlp-fake-news-final.ipynb) · [**LinkedIn**](https://www.linkedin.com/in/abdelrahman-gamal-zayed/)

</div>

---

## Overview

Veridica is an end-to-end fake news detection system that:

- Trains and evaluates **9 NLP architectures** side-by-side
- Supports **English** (WELFake, ~72k articles) and **Arabic** (Arabic Fake News dataset, ~100k articles)
- Deploys as an interactive **Streamlit web app** with a live article scanner
- Uses real trained models for inference — with an LLM fallback when models are not present

| Language | Best Model | Accuracy | F1-Score |
|----------|-----------|----------|----------|
| English  | DistilBERT | 97.87% | 97.61% |
| English  | Logistic Regression | 96.16% | 95.74% |
| English  | BiLSTM | 95.81% | 95.41% |
| English  | XGBoost | 95.78% | 95.36% |
| English  | CNN-LSTM | 94.49% | 93.92% |
| Arabic   | AraBERTv2 | 85.27% | 85.75% |
| Arabic   | Random Forest | 80.90% | 81.35% |
| Arabic   | BiLSTM | 78.69% | 78.99% |
| Arabic   | Logistic Regression | 78.06% | 78.10% |
| Arabic   | CNN-LSTM | 76.27% | 77.35% |

---

## Architecture

```
Input Text
    │
    ▼
Language Detection (Arabic Unicode heuristic)
    │
    ├── English ──► Lemmatization + Stopword Removal
    │                       │
    │           ┌───────────┴───────────┐
    │           ▼                       ▼
    │      TF-IDF Features       Sequence Tokens (25k vocab)
    │      LR / XGBoost          BiLSTM / CNN-LSTM
    │
    └── Arabic ───► Unicode Normalization + Diacritic Removal
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
           TF-IDF Features       Sequence Tokens
           LR / Random Forest    BiLSTM / CNN-LSTM
    │
    ▼
Ensemble Average → Threshold (0.65 / 0.35) → REAL / FAKE / UNCERTAIN
```

---

## Project Structure

```
veridica/
│
├── app.py                    # Streamlit application (single file)
├── imgs_b64.py               # Notebook charts embedded as base64
├── NOTEBOOK_SAVE_CODE.py     # Code to add to Kaggle notebook to save models
├── requirements.txt          # Python dependencies
│
├── notebooks/
│   └── nlp-fake-news-final.ipynb   # Training notebook
│
├── assets/                   # Raw chart images from notebook output
│   ├── label_dist.png
│   ├── wordcloud.png
│   ├── roc_curve.png
│   └── ...
│
├── models/                   # ← Put your model files here (git-ignored)
│   └── .gitkeep              # Placeholder — see "Getting the Models" below
│
└── .streamlit/
    ├── config.toml           # Dark theme settings
    └── secrets.toml.example  # Copy → secrets.toml, add your API key
```

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/TensorSquad/veridica.git
cd veridica
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Minimal install** (no TensorFlow, classical models only):
> ```bash
> pip install streamlit plotly anthropic joblib scikit-learn xgboost nltk
> ```

### 4. Download NLTK data

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### 5. Get the models

See **Getting the Models** below, then place files in `models/`.

### 6. Run

```bash
streamlit run app.py
```

Open **http://localhost:8501**

The nav bar shows **LOCAL MODELS** when real inference is active.

---

## Getting the Models

The trained model files are not tracked by git (too large).
You have two options:

### Option A — Re-run the notebook (recommended)

1. Open `notebooks/nlp-fake-news-final.ipynb` on [Kaggle](https://kaggle.com)
2. Run all cells
3. Add a new cell at the end with the code from `NOTEBOOK_SAVE_CODE.py`
4. Run it — models save to `/kaggle/working/models/`
5. Download from **Kaggle → Output tab**
6. Place all files in your local `models/` folder

### Option B — Download from releases

Pre-trained classical models (LR, XGBoost, TF-IDF) are available in [GitHub Releases](https://github.com/TensorSquad/veridica/releases).
Download `models-classical.zip`, extract into `models/`.

### Expected files in `models/`

| File | Size (approx) |
|------|--------------|
| `tfidf_english.pkl` | ~25 MB |
| `lr_english.pkl` | ~1 MB |
| `xgb_english.pkl` | ~5 MB |
| `tfidf_arabic.pkl` | ~20 MB |
| `lr_arabic.pkl` | ~1 MB |
| `rf_arabic.pkl` | ~15 MB |
| `tokenizer_en.pkl` | ~2 MB |
| `tokenizer_ar.pkl` | ~2 MB |
| `bilstm_en.keras` | ~25 MB |
| `cnn_en.keras` | ~30 MB |
| `bilstm_ar.keras` | ~25 MB |
| `cnn_ar.keras` | ~30 MB |

---

## Deploy

### Streamlit Community Cloud (free, recommended)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo → `app.py`
4. **Advanced settings → Secrets**, paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Deploy — live in ~60 seconds

> ⚠️ Streamlit Cloud has a 1 GB app limit.
> Push only the 6 classical `.pkl` files (≈70 MB total).
> The Keras models (~120 MB total) can also be pushed if under the limit.
> Do NOT push DistilBERT or AraBERTv2 weights.

### Hugging Face Spaces (alternative)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. SDK: **Streamlit**
3. Upload all files
4. Add `ANTHROPIC_API_KEY` under **Settings → Variables and secrets**

---

## Datasets

| Dataset | Language | Size | Source |
|---------|----------|------|--------|
| WELFake | English | ~72k articles | [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |
| Arabic Fake News | Arabic | ~100k articles | [Kaggle](https://www.kaggle.com/datasets/mtwalaa/arabic-fake-news-dataset) |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data & Training | Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow/Keras |
| Transformers | HuggingFace Transformers (DistilBERT, AraBERTv2) |
| NLP Preprocessing | NLTK (EN), Unicode normalization (AR) |
| Web App | Streamlit, Plotly |
| LLM Fallback | Anthropic Claude API |

---

## Author

**Abdelrahman Gamal Zayed**  
AI Engineering Student  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/abdelrahman-gamal-zayed/)
[![GitHub](https://img.shields.io/badge/GitHub-TensorSquad-181717?style=flat&logo=github)](https://github.com/TensorSquad)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
