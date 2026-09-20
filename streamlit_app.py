# ─── FILE: streamlit_app.py ──────────────────────────────────────────────────
# PneumoScan — AI-Powered Lung Sound Classifier
# Corrected for Streamlit Cloud deployment
# Changes from original:
#   1. tensorflow → tflite-runtime compatible import (fixes 600MB crash)
#   2. pymongo import kept (added to requirements.txt)
#   3. All UI code unchanged — identical to original

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import base64
import datetime
import io
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from scipy.ndimage import zoom

# ── TFLite import — works on both Streamlit Cloud and local machine ────────────
try:
    # Streamlit Cloud: lightweight tflite-runtime (5 MB)
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    _TFLITE_SOURCE = "tflite_runtime"
except ImportError:
    try:
        # Local machine: full TensorFlow
        import tensorflow as tf
        TFLiteInterpreter = tf.lite.Interpreter
        _TFLITE_SOURCE = "tensorflow"
    except ImportError:
        TFLiteInterpreter = None
        _TFLITE_SOURCE = "none"

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
SAMPLE_RATE = 8000
N_MELS      = 40
MEL_WIDTH   = 128
FMAX        = 4000
MODEL_PATH  = "model/sonicdaig.tflite"
LUNG_IMG    = ROOT / "img" / "lung.png"
MONGO_DB         = "pneumoscan"
MONGO_COLLECTION = "patients"


def resolve_mongo_uri() -> str:
    """Read the Atlas URI from Streamlit secrets or the MONGO_URI env var."""
    try:
        if "mongo" in st.secrets and st.secrets["mongo"].get("uri"):
            return str(st.secrets["mongo"]["uri"]).strip()
        if "MONGO_URI" in st.secrets:
            return str(st.secrets["MONGO_URI"]).strip()
    except Exception:
        pass
    return (os.environ.get("MONGO_URI") or "").strip()


def image_data_uri(path: Path) -> str:
    """Embed a local PNG so CSS background-image works inside Streamlit."""
    return (
        f"data:image/png;base64,"
        f"{base64.b64encode(path.read_bytes()).decode('ascii')}"
    )


LUNG_URI = image_data_uri(LUNG_IMG) if LUNG_IMG.exists() else ""

LABELS       = ["Normal", "Crackle", "Wheeze"]
LABEL_DETAIL = [
    "Healthy respiratory pattern",
    "Possible pneumonia / COPD pattern",
    "Possible asthma / URTI pattern",
]
COLORS = ["#1A7A5C", "#C44536", "#C47A1A"]
TINTS  = ["#E7F5EF", "#FBECEA", "#FBF1E3"]
ADVICE = [
    "Lungs sound within a healthy range for this recording. Continue routine monitoring.",
    "Crackle pattern detected. Refer the patient to the nearest PHC for clinical review.",
    "Wheeze pattern detected. Refer the patient to the nearest PHC for clinical review.",
]
HINDI = ["सामान्य", "क्रैकल — रेफर करें", "व्हीज — रेफर करें"]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan",
    page_icon="🫁",
    layout="centered"
)

# ── CSS — completely unchanged from original ───────────────────────────────────
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Sora:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {{
  --ink: #12332C;
  --soft: #5E746C;
  --line: #D5E4DE;
  --teal: #0E6B63;
  --teal-deep: #0A4F4A;
  --bg: #F3F8F5;
  --paper: #FFFFFF;
  --warn: #8A5A12;
  --warn-bg: #F8F0DF;
  --lung: url("{LUNG_URI}");
}}

html, body, [class*="css"] {{
  font-family: "Sora", sans-serif;
  color: var(--ink);
}}

.stApp {{
  background-color: var(--bg);
  background-image:
    linear-gradient(180deg, rgba(243,248,245,.72) 0%, rgba(243,248,245,.94) 55%, var(--bg) 100%),
    var(--lung),
    radial-gradient(920px 420px at 50% -180px, #C9E6DB 0%, transparent 58%),
    linear-gradient(180deg, #EAF4EF 0%, var(--bg) 42%, #EEF3F0 100%);
  background-repeat: no-repeat, no-repeat, no-repeat, no-repeat;
  background-position: center top, right -8% top 4%, center top, center top;
  background-size: auto, min(58vw, 520px), auto, auto;
  background-attachment: fixed, fixed, scroll, scroll;
}}

.block-container {{
  max-width: 760px;
  padding-top: 1.1rem;
  padding-bottom: 3.5rem;
}}

#MainMenu, footer, header {{ visibility: hidden; }}

.hero {{
  position: relative;
  overflow: hidden;
  border-radius: 22px;
  min-height: clamp(220px, 38vw, 320px);
  margin: 0 0 1.1rem;
  padding: 1.7rem 1.4rem 1.55rem;
  color: #EAF7F3;
  text-align: left;
  isolation: isolate;
  animation: rise 700ms ease both;
  box-shadow: 0 18px 40px rgba(8, 40, 36, .18);
}}
.hero::before {{
  content: "";
  position: absolute;
  inset: 0;
  z-index: -2;
  background:
    linear-gradient(105deg, rgba(4,18,22,.92) 0%, rgba(6,28,34,.72) 42%, rgba(6,28,34,.28) 100%),
    var(--lung);
  background-repeat: no-repeat, no-repeat;
  background-position: center, right 8% center;
  background-size: cover, auto 118%;
}}
.hero::after {{
  content: "";
  position: absolute;
  inset: 0;
  z-index: -1;
  background: radial-gradient(420px 220px at 78% 55%, rgba(56,180,220,.18), transparent 70%);
  pointer-events: none;
}}
.hero-mark {{
  width: 44px; height: 44px; margin: 0 0 .9rem;
  border-radius: 12px;
  border: 1px solid rgba(180, 230, 220, .28);
  background:
    linear-gradient(145deg, rgba(255,255,255,.12), rgba(255,255,255,.02)),
    var(--lung);
  background-repeat: no-repeat, no-repeat;
  background-position: center, center;
  background-size: auto, cover;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.06);
}}
.hero h1 {{
  font-family: "Fraunces", serif;
  font-size: clamp(2.5rem, 7vw, 3.45rem);
  letter-spacing: -.045em;
  line-height: 1;
  margin: 0;
  color: #F4FFFB;
  max-width: 11ch;
  text-shadow: 0 8px 28px rgba(0,0,0,.35);
}}
.hero p {{
  max-width: 28rem;
  margin: .75rem 0 0;
  color: rgba(214, 236, 230, .88);
  font-size: 1rem;
  line-height: 1.55;
}}
.hero-meta {{
  display: inline-flex;
  flex-wrap: wrap;
  gap: .85rem 1.1rem;
  margin-top: 1.05rem;
  font-family: "IBM Plex Mono", monospace;
  font-size: .7rem;
  letter-spacing: .05em;
  text-transform: uppercase;
  color: #8FD8CB;
}}

.section {{
  margin-top: 1.35rem;
  animation: rise 800ms ease both;
}}
.section-label {{
  font-family: "IBM Plex Mono", monospace;
  font-size: .7rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--teal);
  margin-bottom: .55rem;
}}
.section h2 {{
  font-family: "Fraunces", serif;
  font-size: 1.45rem;
  letter-spacing: -.02em;
  margin: 0 0 .35rem;
}}
.section .lead {{
  color: var(--soft);
  font-size: .92rem;
  margin: 0 0 .9rem;
  line-height: 1.5;
}}

.panel {{
  background: rgba(255,255,255,.88);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1.1rem 1.15rem 1.2rem;
  backdrop-filter: blur(6px);
}}

.metric-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: .65rem;
  margin: .85rem 0 0;
}}
.metric {{
  background: rgba(247,251,250,.92);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: .7rem .8rem;
}}
.metric small {{
  display: block;
  font-family: "IBM Plex Mono", monospace;
  font-size: .66rem;
  letter-spacing: .06em;
  text-transform: uppercase;
  color: var(--soft);
  margin-bottom: .2rem;
}}
.metric b {{ font-size: 1.05rem; color: var(--ink); }}

.result {{
  border-radius: 18px;
  border: 1px solid;
  padding: 1.2rem 1.25rem;
  margin-top: .4rem;
  animation: rise 500ms ease both;
  backdrop-filter: blur(4px);
}}
.result h3 {{
  font-family: "Fraunces", serif;
  font-size: 1.85rem;
  letter-spacing: -.03em;
  margin: 0;
}}
.result .detail {{
  margin: .25rem 0 0;
  font-size: .95rem;
  opacity: .9;
}}
.result .advice {{
  margin: .85rem 0 0;
  color: var(--soft);
  line-height: 1.55;
  font-size: .9rem;
}}
.result .conf {{
  margin-top: .65rem;
  font-family: "IBM Plex Mono", monospace;
  font-size: .78rem;
  letter-spacing: .03em;
}}

.bar-row {{
  display: flex;
  justify-content: space-between;
  font-size: .84rem;
  margin: .55rem 0 .2rem;
}}
.bar {{
  height: 8px;
  background: #E6EFEC;
  border-radius: 99px;
  overflow: hidden;
}}
.bar i {{
  display: block;
  height: 100%;
  border-radius: 99px;
  transition: width 700ms ease;
}}

.note {{
  margin-top: 1.8rem;
  text-align: center;
  color: var(--soft);
  font-size: .78rem;
  line-height: 1.55;
}}
.disclaimer {{
  margin-top: .2rem;
  background: rgba(248,240,223,.92);
  color: var(--warn);
  border: 1px solid #E7D7B4;
  border-radius: 12px;
  padding: .75rem .9rem;
  font-size: .8rem;
  line-height: 1.5;
}}

div[data-testid="stFileUploader"] section {{
  border: 1.5px dashed #B7D0C7 !important;
  border-radius: 14px !important;
  background: rgba(247,251,250,.9) !important;
}}
div[data-testid="stFileUploader"] section:hover {{
  border-color: var(--teal) !important;
  background: #F0F8F5 !important;
}}

.stButton > button[kind="primary"] {{
  background: var(--teal-deep) !important;
  border: none !important;
  border-radius: 12px !important;
  font-family: "Sora", sans-serif !important;
  font-weight: 600 !important;
  padding: .65rem 1rem !important;
  transition: transform 160ms ease, background 160ms ease !important;
}}
.stButton > button[kind="primary"]:hover {{
  background: var(--teal) !important;
  transform: translateY(-1px);
}}

@keyframes rise {{
  from {{ opacity: 0; transform: translateY(10px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}

@media (max-width: 640px) {{
  .metric-grid {{ grid-template-columns: 1fr; }}
  .hero {{
    text-align: center;
    min-height: 260px;
    padding: 1.45rem 1.1rem 1.35rem;
  }}
  .hero::before {{
    background-position: center, center 20%;
    background-size: cover, auto 100%;
  }}
  .hero-mark {{ margin-left: auto; margin-right: auto; }}
  .hero h1 {{
    font-size: 2.45rem;
    max-width: none;
    margin-left: auto;
    margin-right: auto;
  }}
  .hero p {{ margin-left: auto; margin-right: auto; }}
  .stApp {{
    background-position: center top, center top 12%, center top, center top;
    background-size: auto, min(92vw, 420px), auto, auto;
  }}
}}
</style>
""",
    unsafe_allow_html=True,
)


# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if TFLiteInterpreter is None:
        raise RuntimeError(
            "No TFLite runtime found. Install tflite-runtime or tensorflow."
        )
    interpreter = TFLiteInterpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


# ── MongoDB ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_mongo_collection(uri: str):
    """Connect once per app process to the Atlas PneumoScan database."""
    if not uri:
        raise RuntimeError(
            "MongoDB Atlas URI is missing. Add it to .streamlit/secrets.toml "
            "under [mongo] uri, or set the MONGO_URI environment variable."
        )
    if "localhost" in uri or "127.0.0.1" in uri:
        raise RuntimeError(
            "Local MongoDB URIs are disabled. Use your Atlas cluster connection string."
        )
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    return client[MONGO_DB][MONGO_COLLECTION]


def save_patient_record(document: dict) -> str:
    """Insert a screening record and return the new document id."""
    collection = get_mongo_collection(resolve_mongo_uri())
    result     = collection.insert_one(document)
    return str(result.inserted_id)


# ── Signal processing ──────────────────────────────────────────────────────────
def wav_to_spec(y):
    target = SAMPLE_RATE * 10
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    y = y[:target].astype(np.float32)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        fmax=FMAX, n_fft=512, hop_length=64
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] != MEL_WIDTH:
        mel_db = zoom(mel_db, (1.0, MEL_WIDTH / mel_db.shape[1]))
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db, mel_norm


def classify(interpreter, mel_norm):
    inp_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]
    inp     = mel_norm[np.newaxis, ..., np.newaxis].astype(np.float32)
    interpreter.set_tensor(inp_idx, inp)
    interpreter.invoke()
    return interpreter.get_tensor(out_idx)[0]


def plot_spectrogram(mel_db):
    fig, ax = plt.subplots(figsize=(8, 2.7))
    fig.patch.set_facecolor("#F7FBFA")
    ax.set_facecolor("#F7FBFA")
    img = librosa.display.specshow(
        mel_db,
        sr=SAMPLE_RATE,
        hop_length=64,
        x_axis="time",
        y_axis="mel",
        fmax=FMAX,
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
    ax.set_title(
        "Log-mel spectrogram",
        loc="left", fontsize=11, fontweight="bold", color="#12332C"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel frequency")
    for spine in ax.spines.values():
        spine.set_color("#D5E4DE")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  UI — identical to original, zero changes
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <div class="hero-mark" aria-hidden="true"></div>
  <h1>PneumoScan</h1>
  <p>Lung-sound screening for rural clinics — upload a breathing recording
     and review the model's event estimate.</p>
  <div class="hero-meta">
    <span>ICBHI-trained</span>
    <span>3-class screen</span>
    <span>Clinic decision support</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Load model (crash early if missing) ───────────────────────────────────────
try:
    interpreter = load_model()
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()

# ── MongoDB (non-blocking) ─────────────────────────────────────────────────────
mongo_ok  = False
mongo_uri = resolve_mongo_uri()
try:
    get_mongo_collection(mongo_uri)
    mongo_ok = True
except Exception:
    get_mongo_collection.clear()

# ── Disclaimer ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="disclaimer">Screening aid only — not a medical diagnosis. '
    'Always interpret findings with a qualified clinician.</div>',
    unsafe_allow_html=True,
)

# ── 01 · Patient ──────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="section">
  <div class="section-label">01 · Patient</div>
  <h2>Patient details</h2>
  <p class="lead">Used with this screening for your clinic notes.</p>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    patient_name = st.text_input("Name",     placeholder="e.g. Rahul Kumar")
with col2:
    patient_age  = st.number_input(
        "Age", min_value=0, max_value=120, value=None, placeholder="Years"
    )
with col3:
    patient_loc  = st.text_input("Location", placeholder="e.g. Mirzapur")

# ── 02 · Recording ────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="section">
  <div class="section-label">02 · Recording</div>
  <h2>Upload a lung-sound WAV</h2>
  <p class="lead">Use a quiet room and at least six seconds of steady breathing.</p>
</div>
""",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Choose a WAV file", type=["wav"], label_visibility="collapsed"
)

if uploaded is None:
    st.markdown(
        """
<div class="panel" style="text-align:center;color:#5E746C;padding:1.7rem 1rem;
     position:relative;overflow:hidden">
  <div style="position:absolute;inset:0;opacity:.12;background-image:var(--lung);
       background-repeat:no-repeat;background-position:center;
       background-size:auto 140%;pointer-events:none"></div>
  <div style="position:relative">Drop a WAV recording here to begin screening.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="note">PneumoScan · ICBHI 2017 · For screening use only</p>',
        unsafe_allow_html=True,
    )
    st.stop()

# ── Load + validate audio ─────────────────────────────────────────────────────
audio_bytes = uploaded.getvalue()
try:
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
except Exception as error:
    st.error(f"Could not read audio file: {error}")
    st.stop()

rms          = float(np.sqrt(np.mean(y ** 2)))
duration_sec = len(y) / SAMPLE_RATE

st.audio(audio_bytes, format="audio/wav")
st.markdown(
    f"""
<div class="metric-grid">
  <div class="metric"><small>Duration</small><b>{duration_sec:.1f} s</b></div>
  <div class="metric"><small>Sample rate</small><b>{SAMPLE_RATE} Hz</b></div>
  <div class="metric"><small>Signal level</small><b>{rms:.4f} RMS</b></div>
</div>
""",
    unsafe_allow_html=True,
)

if rms < 0.001:
    st.warning(
        "Recording is very quiet. Results may be unreliable — "
        "re-record closer to the chest."
    )

# ── 03 · Screening ────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="section">
  <div class="section-label">03 · Screening</div>
  <h2>Classification result</h2>
  <p class="lead">The model maps the recording to Normal, Crackle, or Wheeze.</p>
</div>
""",
    unsafe_allow_html=True,
)

if not st.button(
    "Analyse recording", type="primary", use_container_width=True
):
    st.info("Review the player and signal metrics, then run analysis.")
    st.stop()

with st.spinner("Computing spectrogram and running inference…"):
    mel_db, mel_norm = wav_to_spec(y)
    probs = classify(interpreter, mel_norm)

label = int(np.argmax(probs))
conf  = float(probs[label]) * 100
color, tint = COLORS[label], TINTS[label]

# ── Result card ───────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="result" style="background:{tint};border-color:{color}55">
  <h3 style="color:{color}">{LABELS[label]}</h3>
  <p class="detail" style="color:{color}">
    {LABEL_DETAIL[label]} · {HINDI[label]}
  </p>
  <p class="advice">{ADVICE[label]}</p>
  <p class="conf" style="color:{color}">Confidence {conf:.1f}%</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Class probability bars ────────────────────────────────────────────────────
st.markdown(
    '<div class="section-label" style="margin-top:1.2rem">Class probabilities</div>',
    unsafe_allow_html=True,
)
for index, name in enumerate(LABELS):
    pct = float(probs[index]) * 100
    st.markdown(
        f"""
<div class="bar-row"><span>{name}</span><b>{pct:.1f}%</b></div>
<div class="bar">
  <i style="width:{pct}%;background:{COLORS[index]}"></i>
</div>
""",
        unsafe_allow_html=True,
    )

# ── Spectrogram expander ──────────────────────────────────────────────────────
with st.expander("View spectrogram"):
    fig = plot_spectrogram(mel_db)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Clinical alert ────────────────────────────────────────────────────────────
if label != 0:
    st.error(
        f"Abnormal pattern suggested ({LABELS[label]}). "
        "Refer the patient to the nearest Primary Health Centre "
        "for clinical assessment."
    )
else:
    st.success(
        "No dominant adventitious pattern detected in this recording."
    )

# ── Build record dict ─────────────────────────────────────────────────────────
now    = datetime.datetime.now(datetime.timezone.utc)
record = {
    "patient_name":       (patient_name or "").strip() or "Unknown",
    "age":                int(patient_age) if patient_age is not None else None,
    "location":           (patient_loc or "").strip() or None,
    "recording_filename": uploaded.name,
    "duration_sec":       round(duration_sec, 2),
    "sample_rate":        SAMPLE_RATE,
    "signal_rms":         round(rms, 6),
    "result":             LABELS[label],
    "result_hi":          HINDI[label],
    "result_detail":      LABEL_DETAIL[label],
    "confidence":         round(conf, 2),
    "probabilities": {
        name: round(float(probs[i]) * 100, 2)
        for i, name in enumerate(LABELS)
    },
    "advice":      ADVICE[label],
    "created_at":  now,
    "model_source": _TFLITE_SOURCE,
}

# ── Save to MongoDB (if connected) ────────────────────────────────────────────
if mongo_ok:
    try:
        save_patient_record(record)
    except PyMongoError:
        get_mongo_collection.clear()

# ── Session record expander ───────────────────────────────────────────────────
with st.expander("Session record", expanded=True):
    st.code(
        f"Patient   : {record['patient_name']}\n"
        f"Age       : {record['age'] if record['age'] is not None else 'Unknown'}\n"
        f"Location  : {record['location'] or 'Unknown'}\n"
        f"Date (UTC): {now.strftime('%Y-%m-%d %H:%M')}\n"
        f"Result    : {LABELS[label]} ({HINDI[label]})\n"
        f"Confidence: {conf:.1f}%\n"
        f"Advice    : {ADVICE[label]}"
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="note">PneumoScan · Built on ICBHI 2017 · '
    'Screening use only · Not a clinical diagnosis</p>',
    unsafe_allow_html=True,
)
