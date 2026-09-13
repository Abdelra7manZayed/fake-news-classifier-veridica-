"""
Streamlit app: Bilingual (English + Arabic) Fake News Classifier.

This app merges two separate Kaggle notebooks/projects into one product:

  * nlp-mini-en-final.ipynb                    (English / WELFake dataset)
  * nlp-arabic-fake-news-classification.ipynb  (Arabic fake-news dataset)

Both notebooks explored several models (Logistic Regression, XGBoost,
Random Forest, LSTM / BiLSTM / CNN-LSTM / CNN-BiLSTM, and transformer models
such as DistilBERT / AraBERT / AraBERTv2). For the deployed app we use
TF-IDF + Logistic Regression for both languages: it was one of the strongest
baselines in both notebooks, trains in seconds without a GPU, and ships as a
tiny artifact — which is what makes an interactive, retrainable Streamlit
demo practical. See src/model.py and README.md for details and for how to
plug in a heavier saved model later.
"""

import time

import pandas as pd
import streamlit as st

from src.model import is_trained, load_artifacts, predict, train_and_save
from src.preprocessing import detect_language, ensure_nltk_data

st.set_page_config(
    page_title="Fake News Classifier · EN/AR",
    page_icon="📰",
    layout="centered",
)

LANG_LABELS = {"en": "English", "ar": "Arabic (العربية)"}


@st.cache_resource(show_spinner=False)
def _bootstrap_nltk():
    ensure_nltk_data()
    return True


@st.cache_resource(show_spinner=False)
def _cached_artifacts(lang: str, _version: float):
    """`_version` is bumped whenever training finishes, forcing a reload."""
    return load_artifacts(lang)


def get_artifacts(lang: str):
    version = st.session_state.get(f"{lang}_version", 0.0)
    return _cached_artifacts(lang, version)


def bump_version(lang: str):
    st.session_state[f"{lang}_version"] = time.time()
    _cached_artifacts.clear()


def render_predict_tab():
    st.subheader("Check an article")

    trained_langs = [l for l in ("en", "ar") if is_trained(l)]
    if not trained_langs:
        st.info(
            "No trained model found yet. Head over to the **Train** tab, "
            "upload a CSV dataset (English or Arabic), and train a model "
            "in a couple of minutes — then come back here."
        )
        return

    options = ["Auto-detect"] + [LANG_LABELS[l] for l in trained_langs]
    choice = st.radio("Language", options, horizontal=True)

    text = st.text_area(
        "Paste a news headline or article body",
        height=180,
        placeholder="Type or paste text here...",
    )

    if st.button("Analyze", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter some text first.")
            return

        if choice == "Auto-detect":
            lang = detect_language(text)
            if lang not in trained_langs:
                st.error(
                    f"Detected language '{LANG_LABELS.get(lang, lang)}' but no "
                    f"model is trained for it yet. Trained models: "
                    f"{', '.join(LANG_LABELS[l] for l in trained_langs)}."
                )
                return
        else:
            lang = "en" if choice == LANG_LABELS["en"] else "ar"

        artifacts = get_artifacts(lang)
        if artifacts is None:
            st.error("Could not load the model artifacts. Try retraining it in the Train tab.")
            return

        label, confidence, top_words = predict(text, lang, artifacts)
        if label is None:
            st.warning("After cleaning, there was no usable text left to analyze — try a longer passage.")
            return

        st.caption(f"Detected/selected language: **{LANG_LABELS[lang]}**")

        is_fake = str(label).strip().lower() in {"fake", "not credible", "1"}
        if is_fake:
            st.error(f"**Prediction: {label}**  ·  confidence {confidence:.1%}")
        else:
            st.success(f"**Prediction: {label}**  ·  confidence {confidence:.1%}")

        st.progress(min(max(confidence, 0.0), 1.0))

        if top_words:
            st.caption("Words that most pushed the model toward this prediction:")
            st.write(" · ".join(f"`{w}`" for w in top_words))


def render_train_tab():
    st.subheader("Train / retrain a model")
    st.markdown(
        "Upload the raw dataset CSV used in the original notebooks "
        "(e.g. `WELFake_Dataset.csv` for English, or an Arabic fake-news CSV) "
        "and train a fresh TF-IDF + Logistic Regression model right here."
    )

    lang_choice = st.selectbox("Language of this dataset", ["English", "Arabic"])
    lang = "en" if lang_choice == "English" else "ar"

    uploaded = st.file_uploader("Dataset CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df):,}** rows.")
        st.dataframe(df.head(5), use_container_width=True)

        cols = list(df.columns)
        text_col = st.selectbox("Text / article column", cols, index=cols.index("text") if "text" in cols else 0)
        title_col = st.selectbox(
            "Title column (optional, will be concatenated with the text column)",
            ["(none)"] + cols,
            index=(["(none)"] + cols).index("title") if "title" in cols else 0,
        )
        title_col = None if title_col == "(none)" else title_col
        label_col = st.selectbox("Label column", cols, index=cols.index("label") if "label" in cols else 0)

        if st.button(f"Train {lang_choice} model", type="primary"):
            progress_bar = st.progress(0.0)
            status = st.empty()

            def progress_cb(msg, pct):
                progress_bar.progress(min(max(pct, 0.0), 1.0))
                status.write(msg)

            with st.spinner("Training..."):
                artifacts = train_and_save(
                    df,
                    lang=lang,
                    label_col=label_col,
                    text_col=text_col,
                    title_col=title_col,
                    progress_callback=progress_cb,
                )

            bump_version(lang)
            st.success("Model trained and saved.")
            m = artifacts.metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{m['accuracy']:.3f}")
            c2.metric("Precision", f"{m['precision']:.3f}")
            c3.metric("Recall", f"{m['recall']:.3f}")
            c4.metric("F1-Score", f"{m['f1']:.3f}")
            st.caption(f"Trained on {m['n_train']:,} rows · evaluated on {m['n_test']:,} rows.")

    st.divider()
    st.markdown("**Currently trained models:**")
    for l in ("en", "ar"):
        if is_trained(l):
            artifacts = get_artifacts(l)
            m = artifacts.metrics if artifacts else {}
            if m:
                st.write(
                    f"- {LANG_LABELS[l]}: ✅ trained "
                    f"(accuracy {m.get('accuracy', 0):.3f}, f1 {m.get('f1', 0):.3f}, "
                    f"at {m.get('trained_at', 'unknown time')})"
                )
            else:
                st.write(f"- {LANG_LABELS[l]}: ✅ trained")
        else:
            st.write(f"- {LANG_LABELS[l]}: ❌ not trained yet")


def render_about_tab():
    st.subheader("About this project")
    st.markdown(
        """
This app merges **two separate notebooks/projects** into a single bilingual
fake-news classifier:

| | English project | Arabic project |
|---|---|---|
| Notebook | `nlp-mini-en-final.ipynb` | `nlp-arabic-fake-news-classification.ipynb` |
| Dataset | WELFake (title + text, real/fake) | Arabic fake-news dataset (credible / not credible) |
| Preprocessing | URL/HTML/number/punctuation stripping, stopword removal, lemmatization | Arabic normalization, diacritics removal, stopword removal, ISRI stemming |
| Models explored | Logistic Regression, XGBoost, LSTM, CNN-LSTM, BiLSTM, DistilBERT | BiLSTM, CNN-BiLSTM, Logistic Regression, Random Forest, AraBERT, Arabic DistilBERT, AraBERTv2 |
| **Deployed here** | TF-IDF + Logistic Regression | TF-IDF + Logistic Regression |

**Why TF-IDF + Logistic Regression for the deployed app**, out of everything
that was tried in the notebooks: it was a strong, stable baseline in *both*
projects, it trains in seconds to minutes on a normal CPU (no GPU, no
internet access to download transformer weights required), and the saved
model is only a few megabytes — which is what makes a self-serve,
retrainable Streamlit app practical. The heavier deep-learning and
transformer models from the notebooks are documented in `notebooks/` for
reference, and `src/model.py` is written so a saved Keras/BERT checkpoint
can be swapped in later without changing this UI.

**How prediction works:** your text is cleaned with the same pipeline used
during training (language-specific), turned into a TF-IDF vector, and
scored by the corresponding Logistic Regression model. The words shown
under a prediction are the ones with the largest positive contribution
(`tfidf weight × model coefficient`) toward the predicted class.
        """
    )


def main():
    _bootstrap_nltk()
    st.title("📰 Fake News Classifier")
    st.caption("English + Arabic · merged from two notebooks into one app")

    tab_predict, tab_train, tab_about = st.tabs(["🔍 Predict", "🛠️ Train", "ℹ️ About"])
    with tab_predict:
        render_predict_tab()
    with tab_train:
        render_train_tab()
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
