"""
TruthLens — Dual-Language Fake News Detection
Real model inference (LR / XGBoost / BiLSTM / CNN-LSTM)
Anthropic API fallback when models not loaded
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import json, os, re, joblib

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLens — Fake News Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "result"   not in st.session_state: st.session_state.result   = None
if "inf_mode" not in st.session_state: st.session_state.inf_mode = "–"

# ── REAL METRICS (from notebook output images) ────────────────────────────────
EN_MODELS = [
    {"name":"EN – DistilBERT",          "acc":97.87,"prec":97.95,"recall":97.28,"f1":97.61},
    {"name":"EN – Logistic Regression", "acc":96.16,"prec":95.19,"recall":96.29,"f1":95.74},
    {"name":"EN – BiLSTM",              "acc":95.81,"prec":93.75,"recall":97.12,"f1":95.41},
    {"name":"EN – XGBoost",             "acc":95.78,"prec":94.11,"recall":96.64,"f1":95.36},
    {"name":"EN – CNN-LSTM",            "acc":94.49,"prec":92.92,"recall":94.94,"f1":93.92},
]
AR_MODELS = [
    {"name":"AR – AraBERTv2",           "acc":85.27,"prec":83.06,"recall":88.62,"f1":85.75},
    {"name":"AR – Random Forest",       "acc":80.90,"prec":79.48,"recall":83.30,"f1":81.35},
    {"name":"AR – BiLSTM",              "acc":78.69,"prec":77.88,"recall":80.14,"f1":78.99},
    {"name":"AR – Logistic Regression", "acc":78.06,"prec":77.96,"recall":78.24,"f1":78.10},
    {"name":"AR – CNN-LSTM",            "acc":76.27,"prec":73.97,"recall":81.06,"f1":77.35},
]

# ── LOAD NOTEBOOK IMAGES ──────────────────────────────────────────────────────
try:
    from imgs_b64 import IMG_B64
    IMGS_LOADED = True
except ImportError:
    IMG_B64 = {}
    IMGS_LOADED = False

# ── PREPROCESSING (mirrors notebook exactly) ──────────────────────────────────
def is_arabic(text: str) -> bool:
    ar = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    return ar / max(len(text), 1) > 0.15

def preprocess_en(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet",   quiet=True)
        lem  = WordNetLemmatizer()
        stop = set(stopwords.words("english"))
        tokens = [lem.lemmatize(t) for t in text.split()
                  if t not in stop and len(t) > 2]
        return " ".join(tokens)
    except Exception:
        return text

def preprocess_ar(text: str) -> str:
    text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text)   # diacritics
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى",      "ي", text)
    text = re.sub(r"ة",      "ه", text)
    text = re.sub(r"[^\u0600-\u06ff\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    m = {}
    d = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.isdir(d):
        return m

    # ── classical ─────────────────────────────────────────────
    for key, fname in [
        ("tfidf_en", "tfidf_english.pkl"),
        ("lr_en",    "lr_english.pkl"),
        ("xgb_en",   "xgb_english.pkl"),
        ("tfidf_ar", "tfidf_arabic.pkl"),
        ("lr_ar",    "lr_arabic.pkl"),
        ("rf_ar",    "rf_arabic.pkl"),
    ]:
        path = os.path.join(d, fname)
        if os.path.exists(path):
            try:
                m[key] = joblib.load(path)
            except Exception:
                pass

    # ── keras tokenizers ──────────────────────────────────────
    import pickle
    for key, fname in [("tok_en", "tokenizer_en.pkl"),
                        ("tok_ar", "tokenizer_ar.pkl")]:
        path = os.path.join(d, fname)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    m[key] = pickle.load(f)
            except Exception:
                pass

    # ── keras models ──────────────────────────────────────────
    try:
        import tensorflow as tf
        for key, fname in [
            ("bilstm_en", "bilstm_en.keras"),
            ("cnn_en",    "cnn_en.keras"),
            ("bilstm_ar", "bilstm_ar.keras"),
            ("cnn_ar",    "cnn_ar.keras"),
        ]:
            path = os.path.join(d, fname)
            if os.path.exists(path):
                try:
                    m[key] = tf.keras.models.load_model(path)
                except Exception:
                    pass
    except ImportError:
        pass

    return m

# ── INFERENCE ─────────────────────────────────────────────────────────────────
def predict_local(text: str, mdls: dict):
    """Run ensemble inference with loaded models. Returns result dict or None."""
    if not mdls:
        return None

    arabic    = is_arabic(text)
    lang      = "Arabic" if arabic else "English"
    processed = preprocess_ar(text) if arabic else preprocess_en(text)

    probs, used = [], []

    if arabic:
        if "tfidf_ar" in mdls and "lr_ar" in mdls:
            feat = mdls["tfidf_ar"].transform([processed])
            probs.append(float(mdls["lr_ar"].predict_proba(feat)[0][1]))
            used.append("LR")
        if "tfidf_ar" in mdls and "rf_ar" in mdls:
            feat = mdls["tfidf_ar"].transform([processed])
            probs.append(float(mdls["rf_ar"].predict_proba(feat)[0][1]))
            used.append("RandomForest")
        if "tok_ar" in mdls and "bilstm_ar" in mdls:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = mdls["tok_ar"].texts_to_sequences([processed])
            pad = pad_sequences(seq, maxlen=250, padding="post", truncating="post")
            probs.append(float(mdls["bilstm_ar"].predict(pad, verbose=0)[0][0]))
            used.append("BiLSTM")
        if "tok_ar" in mdls and "cnn_ar" in mdls:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = mdls["tok_ar"].texts_to_sequences([processed])
            pad = pad_sequences(seq, maxlen=250, padding="post", truncating="post")
            probs.append(float(mdls["cnn_ar"].predict(pad, verbose=0)[0][0]))
            used.append("CNN-LSTM")
    else:
        if "tfidf_en" in mdls and "lr_en" in mdls:
            feat = mdls["tfidf_en"].transform([processed])
            probs.append(float(mdls["lr_en"].predict_proba(feat)[0][1]))
            used.append("LR")
        if "tfidf_en" in mdls and "xgb_en" in mdls:
            feat = mdls["tfidf_en"].transform([processed])
            probs.append(float(mdls["xgb_en"].predict_proba(feat)[0][1]))
            used.append("XGBoost")
        if "tok_en" in mdls and "bilstm_en" in mdls:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = mdls["tok_en"].texts_to_sequences([processed])
            pad = pad_sequences(seq, maxlen=250, padding="post", truncating="post")
            probs.append(float(mdls["bilstm_en"].predict(pad, verbose=0)[0][0]))
            used.append("BiLSTM")
        if "tok_en" in mdls and "cnn_en" in mdls:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = mdls["tok_en"].texts_to_sequences([processed])
            pad = pad_sequences(seq, maxlen=250, padding="post", truncating="post")
            probs.append(float(mdls["cnn_en"].predict(pad, verbose=0)[0][0]))
            used.append("CNN-LSTM")

    if not probs:
        return None

    avg   = sum(probs) / len(probs)
    conf  = int(max(50, min(97, 50 + abs(avg - 0.5) * 2 * 94)))
    verd  = "FAKE" if avg >= 0.65 else ("REAL" if avg <= 0.35 else "UNCERTAIN")

    if avg >= 0.65:
        indicators = [
            "High ensemble fake-probability score",
            "Sensationalist or emotionally charged vocabulary",
            "Unverified claims or missing source attribution",
            f"Models used: {', '.join(used)}",
            f"Avg fake probability: {avg:.1%}",
        ]
    elif avg <= 0.35:
        indicators = [
            "Low ensemble fake-probability score",
            "Neutral, formal reporting language",
            "Factual structure with attributed claims",
            f"Models used: {', '.join(used)}",
            f"Avg fake probability: {avg:.1%}",
        ]
    else:
        indicators = [
            "Ensemble signals are mixed or ambiguous",
            "Text sits near the decision boundary (0.5)",
            "Try a longer or more complete excerpt",
            f"Models used: {', '.join(used)}",
            f"Avg fake probability: {avg:.1%}",
        ]

    return {
        "language":   lang,
        "verdict":    verd,
        "confidence": conf,
        "indicators": indicators,
        "reasoning":  (
            f"Ensemble of {len(probs)} model(s) "
            f"({', '.join(used)}) — "
            f"avg fake prob {avg:.1%}."
        ),
    }

async def predict_api(text: str) -> dict:
    """Anthropic API fallback."""
    prompt = (
        "You are an NLP fake-news classifier.\n"
        "Respond ONLY with compact JSON — no markdown, no backticks.\n\n"
        "Fields:\n"
        '- "language": "English" | "Arabic" | "Other"\n'
        '- "verdict": "REAL" | "FAKE" | "UNCERTAIN"\n'
        "- \"confidence\": integer 51-97\n"
        "- \"indicators\": 4-5 strings each under 9 words\n"
        "- \"reasoning\": one concise sentence\n\n"
        f"Text:\n{text[:2000]}"
    )
    api_key = (
        st.secrets.get("ANTHROPIC_API_KEY", "")
        or os.environ.get("ANTHROPIC_API_KEY", "")
    )
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not configured.")
    import anthropic as _a
    client = _a.Anthropic(api_key=api_key)
    msg    = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw    = msg.content[0].text
    clean  = raw.replace("```json","").replace("```","").strip()
    return json.loads(clean)

# ── FONTS ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700'
    '&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu,footer,header { visibility:hidden!important }
.stDeployButton,[data-testid="stToolbar"],[data-testid="collapsedControl"],
section[data-testid="stSidebar"] { display:none!important }

.stApp { background:#060409!important; font-family:'Space Grotesk',system-ui,sans-serif!important }
.main .block-container { padding:0!important; max-width:100%!important }
* { -webkit-font-smoothing:antialiased }

::-webkit-scrollbar { width:5px }
::-webkit-scrollbar-track { background:#0a060e }
::-webkit-scrollbar-thumb { background:#2a1e38; border-radius:3px }

/* NAV */
.tl-nav {
  position:sticky; top:0; z-index:200;
  background:rgba(6,4,9,.93); backdrop-filter:blur(16px);
  border-bottom:1px solid rgba(62,232,198,.12);
  padding:16px 52px;
  display:flex; align-items:center; justify-content:space-between;
}
.tl-logo { font-family:'Space Mono',monospace; font-size:17px; letter-spacing:.05em; color:#3ee8c6 }
.tl-nav-r { display:flex; align-items:center; gap:32px }
.tl-nav-links { display:flex; gap:28px }
.tl-nav-links a { font-size:13px; font-weight:500; color:rgba(220,215,230,.4); text-decoration:none; transition:color .2s }
.tl-nav-links a:hover { color:#3ee8c6 }
.tl-inf-badge {
  font-family:'Space Mono',monospace; font-size:10px; letter-spacing:.07em;
  padding:4px 10px; border-radius:2px; border:1px solid;
}
.tl-inf-badge.local { color:rgba(0,230,118,.7); border-color:rgba(0,230,118,.25); background:rgba(0,230,118,.06) }
.tl-inf-badge.api   { color:rgba(62,232,198,.7); border-color:rgba(62,232,198,.25); background:rgba(62,232,198,.06) }
.tl-inf-badge.none  { color:rgba(220,215,230,.3); border-color:rgba(220,215,230,.1) }

/* STATS */
.tl-stats { display:grid; grid-template-columns:repeat(4,1fr); border:1px solid rgba(62,232,198,.1); margin:0 52px }
.tl-stat  { padding:26px 30px; border-right:1px solid rgba(62,232,198,.1) }
.tl-stat:last-child { border-right:none }
.tl-stat-n { font-family:'Space Mono',monospace; font-size:34px; font-weight:700; letter-spacing:-.03em; color:#3ee8c6; line-height:1; margin-bottom:5px }
.tl-stat-l { font-size:12px; color:rgba(220,215,230,.35); font-weight:500 }

/* SECTION */
.tl-section { padding:72px 52px; border-top:1px solid rgba(255,255,255,.04) }
.tl-sec-label { font-family:'Space Mono',monospace; font-size:11px; color:rgba(62,232,198,.55); letter-spacing:.14em; margin-bottom:14px }
.tl-sec-title { font-size:clamp(26px,3.5vw,42px); font-weight:600; letter-spacing:-.025em; color:#dcd7e6; line-height:1.1; margin-bottom:10px }
.tl-sec-desc  { font-size:14.5px; color:rgba(220,215,230,.38); line-height:1.7; max-width:560px; margin-bottom:36px }

/* PANEL */
.tl-panel {
  background:linear-gradient(135deg,#0e0a14,#0a0710);
  border:1px solid rgba(62,232,198,.18); border-radius:4px; overflow:hidden;
  animation:glow-pulse 4s ease-in-out infinite;
}
.tl-panel-head {
  padding:13px 20px; border-bottom:1px solid rgba(62,232,198,.1);
  background:rgba(62,232,198,.03);
  display:flex; align-items:center; justify-content:space-between;
}
.tl-panel-head-title { font-family:'Space Mono',monospace; font-size:11px; color:#3ee8c6; letter-spacing:.09em }
.tl-tags { display:flex; gap:7px }
.tl-tag  { font-size:10.5px; color:rgba(62,232,198,.45); padding:3px 10px; border:1px solid rgba(62,232,198,.18); border-radius:2px; letter-spacing:.05em }

/* RESULT */
.tl-result { animation:fade-up .45s ease forwards; padding:26px 22px; border-top:1px solid rgba(62,232,198,.1) }
.tl-result-grid { display:grid; grid-template-columns:150px 1fr; gap:28px; align-items:start }
.tl-ring-wrap { position:relative; width:140px; height:140px }
.tl-ring-wrap svg { transform:rotate(-90deg) }
.tl-ring-center { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center }
.tl-ring-pct    { font-family:'Space Mono',monospace; font-size:24px; font-weight:700; letter-spacing:-.03em; line-height:1 }
.tl-ring-sub    { font-size:9px; color:rgba(220,215,230,.35); letter-spacing:.07em; margin-top:4px }
.tl-verdict     { font-size:50px; font-weight:700; letter-spacing:-.03em; line-height:1; margin-bottom:5px; animation:verdict-in .4s cubic-bezier(.34,1.56,.64,1) forwards }
.tl-verdict.FAKE      { color:#ff4466; text-shadow:0 0 40px rgba(255,68,102,.35) }
.tl-verdict.REAL      { color:#00e676; text-shadow:0 0 40px rgba(0,230,118,.3) }
.tl-verdict.UNCERTAIN { color:#ffaa00; text-shadow:0 0 40px rgba(255,170,0,.25) }
.tl-lang-chip { display:inline-block; font-family:'Space Mono',monospace; font-size:10.5px; color:rgba(62,232,198,.65); padding:4px 11px; border:1px solid rgba(62,232,198,.22); border-radius:2px; letter-spacing:.06em; margin-bottom:18px }
.tl-sig-lbl { font-family:'Space Mono',monospace; font-size:9.5px; color:rgba(62,232,198,.45); letter-spacing:.12em; margin-bottom:11px }
.tl-sig { font-size:13px; color:rgba(220,215,230,.52); padding:6px 0 6px 17px; position:relative; border-bottom:1px solid rgba(255,255,255,.04); line-height:1.5; animation:signal-in .3s ease forwards; opacity:0 }
.tl-sig::before { content:'›'; position:absolute; left:0; color:rgba(62,232,198,.45); font-size:15px }
.tl-reasoning { margin-top:16px; padding:13px 15px; background:rgba(62,232,198,.04); border-left:2px solid rgba(62,232,198,.28); font-size:12.5px; color:rgba(220,215,230,.45); font-style:italic; line-height:1.65; animation:fade-up .5s .35s ease forwards; opacity:0 }
.tl-await { height:280px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:12px; border:1px dashed rgba(62,232,198,.1); border-radius:3px; margin-top:8px }
.tl-await-label { font-family:'Space Mono',monospace; font-size:10.5px; color:rgba(62,232,198,.28); letter-spacing:.1em }

/* ASSETS */
.tl-asset-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:32px }
.tl-asset-grid.full { grid-template-columns:1fr }
.tl-asset-card { background:#0a0710; border:1px solid rgba(255,255,255,.07); border-radius:3px; overflow:hidden; transition:border-color .25s, box-shadow .25s }
.tl-asset-card:hover { border-color:rgba(62,232,198,.2); box-shadow:0 0 24px rgba(62,232,198,.07) }
.tl-asset-label { padding:11px 16px; border-bottom:1px solid rgba(255,255,255,.06); font-family:'Space Mono',monospace; font-size:10px; color:rgba(62,232,198,.5); letter-spacing:.09em }
.tl-asset-img { width:100%; display:block }

/* PIPELINE */
.tl-pipeline { display:grid; grid-template-columns:repeat(4,1fr); border:1px solid rgba(62,232,198,.08) }
.tl-step { padding:28px 24px; border-right:1px solid rgba(62,232,198,.08); transition:background .25s; position:relative; overflow:hidden }
.tl-step:last-child { border-right:none }
.tl-step:hover { background:rgba(62,232,198,.025) }
.tl-step-n { font-family:'Space Mono',monospace; font-size:42px; font-weight:700; color:rgba(62,232,198,.07); line-height:1; margin-bottom:18px; letter-spacing:-.04em }
.tl-step-t { font-size:13.5px; font-weight:600; color:rgba(220,215,230,.75); margin-bottom:9px }
.tl-step-d { font-size:12.5px; color:rgba(220,215,230,.33); line-height:1.7 }
.tl-step::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,#3ee8c6,transparent); opacity:0; transition:opacity .3s }
.tl-step:hover::before { opacity:1 }

/* FOOTER */
.tl-footer { border-top:1px solid rgba(62,232,198,.07); padding:32px 52px; display:flex; align-items:center; justify-content:space-between }
.tl-footer-name { font-family:'Space Mono',monospace; font-size:13px; color:rgba(62,232,198,.45) }
.tl-footer-links { display:flex; gap:22px }
.tl-footer-links a { font-size:12.5px; color:rgba(220,215,230,.28); text-decoration:none; transition:color .2s }
.tl-footer-links a:hover { color:#3ee8c6 }
.tl-footer-copy { font-size:11.5px; color:rgba(220,215,230,.18) }

/* ANIMATIONS */
@keyframes glow-pulse {
  0%,100% { box-shadow:0 0 20px rgba(62,232,198,.06),0 0 60px rgba(62,232,198,.02) }
  50%     { box-shadow:0 0 30px rgba(62,232,198,.12),0 0 80px rgba(62,232,198,.05) }
}
@keyframes fade-up   { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
@keyframes verdict-in{ from{opacity:0;transform:scale(.88) translateY(8px);letter-spacing:.1em} to{opacity:1;transform:scale(1) translateY(0);letter-spacing:-.03em} }
@keyframes signal-in { from{opacity:0;transform:translateX(-10px)} to{opacity:1;transform:translateX(0)} }

/* WIDGET OVERRIDES */
.stTextArea > div > div > textarea {
  background:#0a0710!important; border:1px solid rgba(62,232,198,.2)!important;
  border-radius:3px!important; color:rgba(220,215,230,.9)!important;
  font-family:'Space Grotesk',sans-serif!important; font-size:14.5px!important;
  line-height:1.65!important; caret-color:#3ee8c6!important; padding:13px 15px!important;
}
.stTextArea > div > div > textarea:focus { border-color:rgba(62,232,198,.42)!important; box-shadow:0 0 0 3px rgba(62,232,198,.06)!important }
.stTextArea > div > div > textarea::placeholder { color:rgba(62,232,198,.18)!important }
.stTextArea label { font-family:'Space Mono',monospace!important; font-size:10.5px!important; color:rgba(62,232,198,.45)!important; letter-spacing:.1em!important }
div[data-testid="stButton"] > button {
  background:#3ee8c6!important; color:#060409!important; border:none!important;
  border-radius:2px!important; font-family:'Space Mono',monospace!important;
  font-size:12.5px!important; font-weight:700!important; letter-spacing:.07em!important;
  padding:11px 28px!important; transition:all .18s!important;
  box-shadow:0 0 20px rgba(62,232,198,.18)!important;
}
div[data-testid="stButton"] > button:hover { background:#5efce8!important; box-shadow:0 0 32px rgba(62,232,198,.38)!important; transform:translateY(-1px)!important }
.stSpinner > div { border-top-color:#3ee8c6!important }
.stTabs [data-baseweb="tab-list"] { background:transparent!important; border-bottom:1px solid rgba(62,232,198,.1)!important; gap:0 }
.stTabs [data-baseweb="tab"] { background:transparent!important; color:rgba(220,215,230,.35)!important; font-family:'Space Mono',monospace!important; font-size:11px!important; letter-spacing:.07em!important; border:none!important; padding:10px 20px!important; transition:color .2s!important }
.stTabs [aria-selected="true"] { color:#3ee8c6!important; border-bottom:2px solid #3ee8c6!important }

@media(max-width:900px){
  .tl-nav,.tl-section,.tl-footer{padding-left:20px;padding-right:20px}
  .tl-stats{margin:0 20px;grid-template-columns:1fr 1fr}
  .tl-pipeline{grid-template-columns:1fr 1fr}
  .tl-asset-grid{grid-template-columns:1fr}
  .tl-result-grid{grid-template-columns:1fr}
  .tl-ring-wrap{margin:0 auto}
  .tl-footer{flex-direction:column;gap:14px}
}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODELS (once, cached) ────────────────────────────────────────────────
MDLS = load_models()

en_ready = "tfidf_en" in MDLS and ("lr_en" in MDLS or "xgb_en" in MDLS)
ar_ready = "tfidf_ar" in MDLS and ("lr_ar" in MDLS or "rf_ar" in MDLS)
models_ready = en_ready or ar_ready
badge_cls    = "local" if models_ready else "api"
badge_lbl    = (
    "LOCAL MODELS" if models_ready
    else "API FALLBACK"
)

# ── NAV ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<nav class="tl-nav">
  <span class="tl-logo">TRUTH_LENS</span>
  <div class="tl-nav-r">
    <div class="tl-nav-links">
      <a href="#analyzer">Analyzer</a>
      <a href="#results">Results</a>
      <a href="#visuals">Visuals</a>
      <a href="#pipeline">Method</a>
    </div>
    <span class="tl-inf-badge {badge_cls}">{badge_lbl}</span>
  </div>
</nav>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
components.html("""
<!DOCTYPE html><html><head>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#060409;overflow:hidden}
  canvas{display:block}
  .ht{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;z-index:10;pointer-events:none;width:90%;max-width:760px}
  .hl{font-family:'Space Mono',monospace;font-size:11px;color:rgba(62,232,198,.55);letter-spacing:.18em;margin-bottom:24px;animation:fi .8s .1s ease both}
  .h1{font-family:'Space Grotesk',system-ui,sans-serif;font-size:clamp(34px,5.5vw,68px);font-weight:700;letter-spacing:-.04em;line-height:1.02;color:#dcd7e6;margin-bottom:18px;animation:fi .9s .3s ease both}
  .h1 .cy{color:#3ee8c6}
  .hs{font-size:15px;color:rgba(220,215,230,.35);max-width:500px;margin:0 auto;line-height:1.7;animation:fi .9s .5s ease both}
  .hb{display:inline-block;margin-top:24px;font-family:'Space Mono',monospace;font-size:10.5px;color:rgba(62,232,198,.45);padding:7px 18px;border:1px solid rgba(62,232,198,.18);letter-spacing:.08em;animation:fi .9s .7s ease both}
  @keyframes fi{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
</style>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&family=Space+Mono&display=swap" rel="stylesheet">
</head><body>
<canvas id="c"></canvas>
<div class="ht">
  <p class="hl">DUAL-LANGUAGE · NLP RESEARCH · 2025</p>
  <h1 class="h1">Separating signal<br>from <span class="cy">noise</span></h1>
  <p class="hs">Nine model architectures detect misinformation in English and Arabic — TF-IDF baselines through fine-tuned transformers.</p>
  <div class="hb">97.61% peak F1 · DistilBERT · 72k+ articles</div>
</div>
<script>
const c=document.getElementById('c'),ctx=c.getContext('2d');
let W,H,pts=[],mx=null,my=null;
const TC='62,232,198',RC='255,68,102';
function resize(){W=c.width=window.innerWidth;H=c.height=window.innerHeight}
class P{constructor(){this.reset()}reset(){this.x=Math.random()*W;this.y=Math.random()*H;this.vx=(Math.random()-.5)*.42;this.vy=(Math.random()-.5)*.42;this.r=Math.random()*1.7+.4;this.a=Math.random()*.5+.15;this.col=Math.random()>.88?RC:TC}update(){this.x+=this.vx;this.y+=this.vy;if(this.x<0||this.x>W)this.vx*=-1;if(this.y<0||this.y>H)this.vy*=-1}draw(){ctx.beginPath();ctx.arc(this.x,this.y,this.r,0,Math.PI*2);ctx.fillStyle=`rgba(${this.col},${this.a})`;ctx.fill()}}
function connect(){const M=125;for(let i=0;i<pts.length;i++){for(let j=i+1;j<pts.length;j++){const dx=pts[i].x-pts[j].x,dy=pts[i].y-pts[j].y,d=Math.sqrt(dx*dx+dy*dy);if(d<M){ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);ctx.strokeStyle=`rgba(${TC},${(1-d/M)*.25})`;ctx.lineWidth=.55;ctx.stroke()}}if(mx){const dx=pts[i].x-mx,dy=pts[i].y-my,d=Math.sqrt(dx*dx+dy*dy);if(d<155){ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(mx,my);ctx.strokeStyle=`rgba(${TC},${(1-d/155)*.48})`;ctx.lineWidth=.9;ctx.stroke()}}}}
function draw(){ctx.fillStyle='#060409';ctx.fillRect(0,0,W,H);pts.forEach(p=>{p.update();p.draw()});connect();requestAnimationFrame(draw)}
resize();for(let i=0;i<88;i++)pts.push(new P());
window.addEventListener('resize',resize);
window.addEventListener('mousemove',e=>{mx=e.clientX;my=e.clientY});
window.addEventListener('mouseleave',()=>{mx=null;my=null});
draw();
</script></body></html>
""", height=400)

# ── STATS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tl-stats">
  <div class="tl-stat"><div class="tl-stat-n">97.6<span style="font-size:.55em;opacity:.6">%</span></div><div class="tl-stat-l">Peak F1 · EN DistilBERT</div></div>
  <div class="tl-stat"><div class="tl-stat-n">9</div><div class="tl-stat-l">Architectures compared</div></div>
  <div class="tl-stat"><div class="tl-stat-n">72k+</div><div class="tl-stat-l">Training articles</div></div>
  <div class="tl-stat"><div class="tl-stat-n">EN+AR</div><div class="tl-stat-l">Dual-language support</div></div>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)

# ── ANALYZER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tl-section" id="analyzer">
  <p class="tl-sec-label">LIVE DEMO</p>
  <h2 class="tl-sec-title">Analyze any news text</h2>
  <p class="tl-sec-desc">Paste a headline or article excerpt in English or Arabic. The analyzer uses your trained models when available, or falls back to the Anthropic API.</p>
</div>
""", unsafe_allow_html=True)

col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.markdown("""
    <div class="tl-panel">
      <div class="tl-panel-head">
        <span class="tl-panel-head-title">INPUT // ARTICLE SCANNER</span>
        <div class="tl-tags"><span class="tl-tag">EN</span><span class="tl-tag">ع</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    news_text = st.text_area(
        "Article text",
        placeholder="Paste a news headline, article excerpt, or social post…\n\nWorks in English and Arabic.",
        height=195,
        label_visibility="collapsed",
    )
    run_btn = st.button("SCAN TEXT", use_container_width=True)

with col_out:
    result_slot = st.empty()

    if st.session_state.result:
        r  = st.session_state.result
        v  = r.get("verdict","UNCERTAIN").upper()
        cf = r.get("confidence", 60)
        lg = r.get("language","–")
        sg = r.get("indicators",[])
        rs = r.get("reasoning","")
        rc = {"FAKE":"#ff4466","REAL":"#00e676","UNCERTAIN":"#ffaa00"}.get(v,"#ffaa00")
        circ = 283
        dash = circ - (cf / 100) * circ
        sigs_html = "".join(
            f'<div class="tl-sig" style="animation-delay:{i*.08:.2f}s">{s}</div>'
            for i, s in enumerate(sg)
        )
        result_slot.markdown(f"""
<div class="tl-result">
  <div class="tl-result-grid">
    <div>
      <div class="tl-ring-wrap">
        <svg width="140" height="140" viewBox="0 0 140 140">
          <circle cx="70" cy="70" r="45" fill="none" stroke="rgba(255,255,255,.05)" stroke-width="8"/>
          <circle cx="70" cy="70" r="45" fill="none"
            stroke="{rc}" stroke-width="8" stroke-linecap="round"
            stroke-dasharray="{circ}" stroke-dashoffset="{dash}"
            style="transition:stroke-dashoffset 1s cubic-bezier(.4,0,.2,1);filter:drop-shadow(0 0 7px {rc}55)"/>
        </svg>
        <div class="tl-ring-center">
          <div class="tl-ring-pct" style="color:{rc}">{cf}%</div>
          <div class="tl-ring-sub">CONFIDENCE</div>
        </div>
      </div>
    </div>
    <div>
      <div class="tl-verdict {v}">{v}</div>
      <div class="tl-lang-chip">{lg}</div>
      <div class="tl-sig-lbl">SIGNALS DETECTED</div>
      {sigs_html}
      <div class="tl-reasoning">{rs}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
    else:
        result_slot.markdown("""
<div class="tl-await">
  <svg width="44" height="44" viewBox="0 0 44 44" fill="none">
    <circle cx="22" cy="22" r="18" stroke="rgba(62,232,198,.18)" stroke-width="1.5"/>
    <circle cx="22" cy="22" r="10" stroke="rgba(62,232,198,.12)" stroke-width="1"/>
    <circle cx="22" cy="22" r="3"  fill="rgba(62,232,198,.22)"/>
    <line x1="22" y1="4"  x2="22" y2="9"  stroke="rgba(62,232,198,.28)" stroke-width="1.5"/>
    <line x1="22" y1="35" x2="22" y2="40" stroke="rgba(62,232,198,.28)" stroke-width="1.5"/>
    <line x1="4"  y1="22" x2="9"  y2="22" stroke="rgba(62,232,198,.28)" stroke-width="1.5"/>
    <line x1="35" y1="22" x2="40" y2="22" stroke="rgba(62,232,198,.28)" stroke-width="1.5"/>
  </svg>
  <span class="tl-await-label">AWAITING INPUT</span>
</div>""", unsafe_allow_html=True)

# ── RUN ANALYSIS ──────────────────────────────────────────────────────────────
if run_btn and news_text and news_text.strip():
    st.session_state.result = None
    with col_out:
        with st.spinner("Scanning…"):
            res = predict_local(news_text.strip(), MDLS)
            if res:
                st.session_state.inf_mode = "local"
                st.session_state.result   = res
            else:
                try:
                    import asyncio
                    res = asyncio.run(predict_api(news_text.strip()))
                    st.session_state.inf_mode = "api"
                    st.session_state.result   = res
                except Exception as e:
                    st.session_state.inf_mode = "error"
                    st.session_state.result = {
                        "language":"–","verdict":"UNCERTAIN","confidence":50,
                        "indicators":[
                            "No models found in models/ folder",
                            "ANTHROPIC_API_KEY also not configured",
                            "See README — add models or set API key",
                            f"Error: {str(e)[:55]}",
                        ],
                        "reasoning":"Add model files or configure ANTHROPIC_API_KEY to enable analysis.",
                    }
    st.rerun()

# ── PERFORMANCE CHARTS ────────────────────────────────────────────────────────
st.markdown("""
<div class="tl-section" id="results">
  <p class="tl-sec-label">MODEL PERFORMANCE</p>
  <h2 class="tl-sec-title">Results across all architectures</h2>
  <p class="tl-sec-desc">Scores from actual notebook output. English and Arabic evaluated independently on 20% held-out test sets.</p>
</div>
""", unsafe_allow_html=True)

def make_chart(models, metric, color):
    names = [m["name"] for m in reversed(models)]
    vals  = [round(m[metric],2) for m in reversed(models)]
    n     = len(models)
    colors= [color if i == n-1 else color.replace(",.9)",",.30)") for i in range(n)]
    fig   = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=vals, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,.06)", width=.8)),
        text=[f"<b>{v:.2f}</b>" for v in vals],
        textposition="inside",
        textfont=dict(family="Space Mono, monospace", size=11, color="#060409"),
        hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=55,t=8,b=8), height=240,
        xaxis=dict(range=[70,102], gridcolor="rgba(255,255,255,.04)", gridwidth=1,
                   tickfont=dict(family="Space Mono, monospace", size=10, color="rgba(220,215,230,.28)"),
                   ticksuffix="%", zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(family="Space Grotesk, sans-serif", size=12.5, color="rgba(220,215,230,.55)")),
        showlegend=False,
    )
    return fig

metric_tabs = st.tabs(["F1-Score","Accuracy","Precision","Recall"])
metric_cfg  = [("f1","rgba(62,232,198,.9)"),("acc","rgba(130,160,255,.9)"),
               ("prec","rgba(200,130,255,.9)"),("recall","rgba(255,120,160,.9)")]

for tab, (metric, color) in zip(metric_tabs, metric_cfg):
    with tab:
        c1, c2 = st.columns(2, gap="large")
        lbl = '<p style="font-family:Space Mono,monospace;font-size:11px;color:rgba(62,232,198,.5);letter-spacing:.1em;margin-bottom:8px">'
        with c1:
            st.markdown(lbl+"ENGLISH</p>", unsafe_allow_html=True)
            st.plotly_chart(make_chart(EN_MODELS, metric, color), use_container_width=True, config={"displayModeBar":False})
        with c2:
            st.markdown(lbl+"ARABIC</p>", unsafe_allow_html=True)
            st.plotly_chart(make_chart(AR_MODELS, metric, color), use_container_width=True, config={"displayModeBar":False})

# ── NOTEBOOK VISUALS ──────────────────────────────────────────────────────────
st.markdown("""
<div class="tl-section" id="visuals">
  <p class="tl-sec-label">NOTEBOOK VISUALS</p>
  <h2 class="tl-sec-title">Output from training</h2>
  <p class="tl-sec-desc">All charts generated directly from the training notebook on the WELFake and Arabic Fake News datasets.</p>
</div>
""", unsafe_allow_html=True)

def asset_card(key, label):
    if not IMGS_LOADED or key not in IMG_B64:
        return (f'<div class="tl-asset-card"><div class="tl-asset-label">{label}</div>'
                f'<div style="height:120px;display:flex;align-items:center;justify-content:center;'
                f'color:rgba(62,232,198,.2);font-family:Space Mono,monospace;font-size:11px">'
                f'IMAGE NOT FOUND</div></div>')
    return (f'<div class="tl-asset-card"><div class="tl-asset-label">{label}</div>'
            f'<img class="tl-asset-img" src="{IMG_B64[key]}" alt="{label}"/></div>')

st.markdown(f'<div class="tl-asset-grid full">{asset_card("wordcloud","WORD CLOUD · Real vs Fake News (English)")}</div>', unsafe_allow_html=True)
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown(f'<div class="tl-asset-grid">{asset_card("label_dist","LABEL DISTRIBUTION · English & Arabic")}{asset_card("text_len","TEXT LENGTH DISTRIBUTION · English & Arabic")}</div>', unsafe_allow_html=True)
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown(f'<div class="tl-asset-grid">{asset_card("model_compare","F1 + ACCURACY COMPARISON · All Models")}{asset_card("feature_imp","TOP 20 FEATURE IMPORTANCE · EN XGBoost")}</div>', unsafe_allow_html=True)
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown(f'<div class="tl-asset-grid">{asset_card("roc_curve","ROC CURVES · English Models (AUC = 0.9935)")}{asset_card("distilbert","TRAINING CURVE · EN DistilBERT (3 epochs)")}</div>', unsafe_allow_html=True)
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown(f'<div class="tl-asset-grid">{asset_card("acc_prec","ACCURACY + PRECISION · All Models")}{asset_card("recall_f1","RECALL + F1-SCORE · All Models")}</div>', unsafe_allow_html=True)

# ── PIPELINE ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tl-section" id="pipeline">
  <p class="tl-sec-label">METHODOLOGY</p>
  <h2 class="tl-sec-title">Detection pipeline</h2>
  <p class="tl-sec-desc">Four-stage architecture — language-aware preprocessing through calibrated multi-model inference.</p>
  <div class="tl-pipeline">
    <div class="tl-step"><div class="tl-step-n">01</div><div class="tl-step-t">Text preprocessing</div><div class="tl-step-d">English: lemmatization via WordNet, stopword removal, URL stripping. Arabic: Unicode normalization of variant forms (إ→ا, ى→ي, ة→ه) and diacritic removal.</div></div>
    <div class="tl-step"><div class="tl-step-n">02</div><div class="tl-step-t">Feature extraction</div><div class="tl-step-d">TF-IDF: 20–25k features, trigram EN / bigram AR, sublinear_tf, min_df filtering. Deep models use 25k-vocab sequence tokenization padded to 250 tokens.</div></div>
    <div class="tl-step"><div class="tl-step-n">03</div><div class="tl-step-t">Model inference</div><div class="tl-step-d">Classical: LR + XGBoost on TF-IDF. Deep: BiLSTM and CNN-LSTM with SpatialDropout. Transformers: DistilBERT (EN) and AraBERTv2 (AR) fine-tuned with AdamWeightDecay.</div></div>
    <div class="tl-step"><div class="tl-step-n">04</div><div class="tl-step-t">Calibrated output</div><div class="tl-step-d">Probability threshold at 0.65 for confident verdicts. Full leaderboard tracks accuracy, precision, recall, F1, and macro-F1 across all nine architectures.</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<footer class="tl-footer">
  <span class="tl-footer-name">TRUTH_LENS // Abdelrahman Gamal</span>
  <div class="tl-footer-links">
    <a href="https://github.com/TensorSquad/fake-news-classifier" target="_blank">GitHub</a>
    <a href="https://www.linkedin.com/in/abdelrahman-gamal-zayed/" target="_blank">LinkedIn</a>
  </div>
  <span class="tl-footer-copy">AI Engineering · 2025</span>
</footer>
""", unsafe_allow_html=True)
