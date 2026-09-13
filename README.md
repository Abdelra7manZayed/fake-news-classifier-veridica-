# 📰 Fake News Classifier — English + Arabic

A single Streamlit app that merges two separate notebooks/projects into one
bilingual fake-news detector:

- **English** — `notebooks/nlp-mini-en-final.ipynb` (WELFake dataset)
- **Arabic** — `notebooks/nlp-arabic-fake-news-classification.ipynb` (Arabic fake-news dataset)

Both notebooks experimented with several models (Logistic Regression,
XGBoost, Random Forest, LSTM / BiLSTM / CNN-LSTM / CNN-BiLSTM, and
transformers such as DistilBERT / AraBERT / AraBERTv2). The app deploys
**TF-IDF + Logistic Regression** for both languages — a strong baseline in
both notebooks that trains in seconds on a CPU and ships as a tiny file,
which is what makes a live, retrainable Streamlit demo practical. The code
is structured so a heavier saved model (Keras `.keras` file or a
`transformers` checkpoint) can be swapped in later — see `src/model.py`.

> والنسخة العربية 👇
>
> ## 📰 مصنّف الأخبار الكاذبة — عربي/إنجليزي
>
> تطبيق Streamlit واحد يدمج مشروعين في أداة واحدة لكشف الأخبار
> الكاذبة بالعربية والإنجليزية: المشروع الإنجليزي (`nlp-mini-en-final.ipynb`
> على بيانات WELFake) والمشروع العربي
> (`nlp-arabic-fake-news-classification.ipynb`). كلا المشروعين جرّبا عدة
> نماذج (Logistic Regression، XGBoost، LSTM/BiLSTM/CNN، ونماذج
> Transformer مثل BERT/AraBERT)، والتطبيق هنا يستخدم **TF-IDF + Logistic
> Regression** لكلتا اللغتين لأنه نموذج قوي وسريع التدريب (بدون GPU) وحجمه
> صغير جدًا، ما يجعل تطبيقًا تفاعليًا وقابلًا لإعادة التدريب أمرًا عمليًا.

## Project structure

```
fake-news-classifier/
├── app.py                  # Streamlit app (Predict / Train / About tabs)
├── train.py                # CLI trainer (train from a CSV without the UI)
├── requirements.txt
├── src/
│   ├── preprocessing.py    # English + Arabic cleaning pipelines
│   └── model.py            # Training, persistence, prediction
├── models/                 # Trained artifacts land here (git-ignored)
├── data/
│   └── sample_texts.json   # A few example headlines for manual testing
├── notebooks/              # The two original source notebooks
└── .streamlit/config.toml  # Theming
```

## Quick start

```bash
git clone <this-repo-url>
cd fake-news-classifier
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

The app opens with **no trained model** (model artifacts are not committed
to the repo — see below). You have two options:

### Option A — train from inside the app (easiest)

1. Open the **🛠️ Train** tab.
2. Choose the language, upload the matching dataset CSV
   (e.g. `WELFake_Dataset.csv` for English, or your Arabic fake-news CSV).
3. Pick the text / title / label columns and click **Train**.
4. Switch to **🔍 Predict** and start testing.

Training happens in-process (TF-IDF + Logistic Regression), so it typically
finishes in well under a minute even on ~50k rows, with no GPU required.

### Option B — train from the command line

```bash
# English (WELFake-style columns: title, text, label)
python train.py --lang en --csv WELFake_Dataset.csv \
    --text-col text --title-col title --label-col label

# Arabic
python train.py --lang ar --csv arabic_fake_news_processed.csv \
    --text-col Article_content --label-col label
```

This writes `models/en_*.joblib` / `models/ar_*.joblib`, which `app.py`
picks up automatically on the next run.

## Why datasets and trained models aren't in this repo

The original datasets (WELFake, Arabic fake-news) were loaded from Kaggle
(`/kaggle/input/...`) inside the notebooks and are not redistributed here
for size/licensing reasons. `.gitignore` also excludes trained model
artifacts, since they're a byproduct you regenerate locally in seconds
rather than binary files that belong in version control. If you want a
"just works, no training step" deployment, train once locally/on Kaggle and
either commit the resulting `models/*.joblib` files to a private repo or
attach them as a release asset.

## Deploying

The app is a standard Streamlit app, so it deploys as-is to
[Streamlit Community Cloud](https://streamlit.io/cloud) (point it at
`app.py`), or any container platform: `Dockerfile` example —

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Pushing this repo to GitHub

```bash
cd fake-news-classifier
git init
git add .
git commit -m "Merge English + Arabic fake news projects into one Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## License

MIT — see `LICENSE`.
