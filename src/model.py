"""
Training, persistence and inference helpers for the bilingual fake-news
classifier.

Deployed model: TF-IDF + Logistic Regression, for BOTH languages.

Why this particular model, out of everything explored in the two source
notebooks (Logistic Regression, XGBoost, LSTM, CNN-LSTM, BiLSTM, CNN-BiLSTM,
Random Forest, DistilBERT / AraBERT / AraBERTv2)?

  * It was one of the strongest, most stable baselines in *both* notebooks.
  * It trains in seconds/minutes on a CPU with no GPU and no internet access,
    which makes it realistic to train and re-train from inside a Streamlit
    app or a small VM.
  * The exported artifacts are tiny (a few MB) compared to the deep-learning
    / transformer checkpoints, which makes the app easy to ship and deploy.

The architecture below is intentionally modular (`MODEL_REGISTRY`) so a
heavier model (e.g. a saved Keras BiLSTM or a fine-tuned BERT/AraBERT
checkpoint exported from the original notebooks) can be dropped in later
without changing the Streamlit UI.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .preprocessing import clean_text, ensure_nltk_data

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# TF-IDF settings, matching the parameters used in each source notebook.
TFIDF_PARAMS = {
    "en": dict(max_features=15000, ngram_range=(1, 3), min_df=2, max_df=0.9,
               sublinear_tf=True, use_idf=True, norm="l2"),
    "ar": dict(max_features=20000, ngram_range=(1, 2), min_df=5, max_df=0.8),
}

LOGREG_PARAMS = {
    "en": dict(max_iter=1000, random_state=42),
    "ar": dict(C=1.0, class_weight="balanced", max_iter=1000, random_state=42),
}

# Human-readable label names. English WELFake-style datasets conventionally
# use 0 = Real, 1 = Fake; the Arabic dataset ships its own string labels
# ("credible" / "not credible") which are used as-is.
DEFAULT_LABEL_NAMES = {
    "en": {0: "Real", 1: "Fake"},
}


@dataclass
class TrainedArtifacts:
    vectorizer: TfidfVectorizer
    model: LogisticRegression
    label_encoder: Optional[LabelEncoder]
    label_names: dict
    metrics: dict = field(default_factory=dict)


def _paths(lang: str):
    return {
        "vectorizer": MODEL_DIR / f"{lang}_tfidf.joblib",
        "model": MODEL_DIR / f"{lang}_logreg.joblib",
        "label_encoder": MODEL_DIR / f"{lang}_label_encoder.joblib",
        "meta": MODEL_DIR / f"{lang}_meta.json",
    }


def is_trained(lang: str) -> bool:
    p = _paths(lang)
    return p["vectorizer"].exists() and p["model"].exists()


def build_combined_text(df: pd.DataFrame, text_col: str, title_col: Optional[str]) -> pd.Series:
    """Recreate the `title + " " + text` concatenation used in both notebooks
    when a separate title column is available."""
    if title_col and title_col in df.columns:
        return (df[title_col].fillna("") + " " + df[text_col].fillna("")).str.strip()
    return df[text_col].fillna("")


def train_and_save(
    df: pd.DataFrame,
    lang: str,
    label_col: str,
    text_col: str,
    title_col: Optional[str] = None,
    test_size: float = 0.2,
    progress_callback=None,
) -> TrainedArtifacts:
    """Clean, vectorize, train and persist a TF-IDF + Logistic Regression
    model for the given language ('en' or 'ar')."""
    ensure_nltk_data()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def report(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    report("Preparing text columns...", 0.05)
    full_text = build_combined_text(df, text_col, title_col)
    labels_raw = df[label_col]

    label_encoder = None
    label_names = DEFAULT_LABEL_NAMES.get(lang, {})
    if pd.api.types.is_numeric_dtype(labels_raw):
        y = labels_raw.astype(int).values
    else:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels_raw.astype(str))
        label_names = {i: cls for i, cls in enumerate(label_encoder.classes_)}

    report(f"Cleaning {len(full_text)} documents ({lang})...", 0.15)
    total = len(full_text)
    cleaned = []
    for i, t in enumerate(full_text):
        cleaned.append(clean_text(t, lang))
        if progress_callback and total > 200 and i % max(1, total // 20) == 0:
            report(f"Cleaning text {i}/{total}...", 0.15 + 0.35 * (i / total))
    cleaned = pd.Series(cleaned)

    mask = cleaned.str.strip() != ""
    cleaned, y = cleaned[mask], y[mask]

    report("Splitting train/test...", 0.55)
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, y, test_size=test_size, random_state=42, stratify=y
    )

    report("Fitting TF-IDF vectorizer...", 0.65)
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS[lang])
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    report("Training Logistic Regression...", 0.8)
    model = LogisticRegression(**LOGREG_PARAMS[lang])
    model.fit(X_train_tfidf, y_train)

    report("Evaluating...", 0.92)
    y_pred = model.predict(X_test_tfidf)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    report("Saving artifacts...", 0.97)
    paths = _paths(lang)
    joblib.dump(vectorizer, paths["vectorizer"])
    joblib.dump(model, paths["model"])
    if label_encoder is not None:
        joblib.dump(label_encoder, paths["label_encoder"])
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump({"label_names": label_names, "metrics": metrics}, f, ensure_ascii=False, indent=2)

    report("Done.", 1.0)
    return TrainedArtifacts(vectorizer, model, label_encoder, label_names, metrics)


def load_artifacts(lang: str) -> Optional[TrainedArtifacts]:
    if not is_trained(lang):
        return None
    paths = _paths(lang)
    vectorizer = joblib.load(paths["vectorizer"])
    model = joblib.load(paths["model"])
    label_encoder = joblib.load(paths["label_encoder"]) if paths["label_encoder"].exists() else None
    meta = {}
    if paths["meta"].exists():
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
    label_names = {int(k): v for k, v in meta.get("label_names", {}).items()} or DEFAULT_LABEL_NAMES.get(lang, {})
    return TrainedArtifacts(vectorizer, model, label_encoder, label_names, meta.get("metrics", {}))


def predict(text: str, lang: str, artifacts: TrainedArtifacts, top_k: int = 8):
    """Return (predicted_label_name, confidence, top_contributing_words)."""
    cleaned = clean_text(text, lang)
    if not cleaned:
        return None, 0.0, []

    vec = artifacts.vectorizer.transform([cleaned])
    proba = artifacts.model.predict_proba(vec)[0]
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])
    label_name = artifacts.label_names.get(pred_idx, str(pred_idx))

    # Top contributing tokens: tfidf weight * logistic-regression coefficient
    # for the predicted class (binary classifier -> single coef vector).
    feature_names = artifacts.vectorizer.get_feature_names_out()
    coefs = artifacts.model.coef_[0]
    row = vec.tocoo()
    contributions = []
    sign = 1 if pred_idx == 1 else -1
    for idx, val in zip(row.col, row.data):
        contributions.append((feature_names[idx], sign * val * coefs[idx]))
    contributions.sort(key=lambda x: x[1], reverse=True)
    top_words = [w for w, s in contributions[:top_k] if s > 0]

    return label_name, confidence, top_words
