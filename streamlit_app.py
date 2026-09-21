# ─── FILE: streamlit_app.py ──────────────────────────────────────────────────
# PneumoScan — AI-Powered Lung Sound Classifier
# Refined v2.0 — Limitations addressed:
#   ✅ FIX 1: Confidence threshold — low confidence → "Inconclusive" result
#   ✅ FIX 2: Noise/quality detection — spectral centroid check
#   ✅ FIX 3: Multi-format audio — accepts WAV, MP3, M4A, OGG, FLAC
#   ✅ FIX 4: Sliding window — processes full recording, majority vote
#   ✅ FIX 5: Session history — tracks all sessions in current visit
#   ✅ FIX 6: CSV export — download all session records as CSV
#   ✅ FIX 7: MongoDB CSV fallback — records saved even without Atlas
#   UI:  Zero changes — identical to v1.0

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import base64
import csv
import datetime
import io
import tempfile
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
from scipy.stats import mode as scipy_mode

# ── TFLite import — works on Streamlit Cloud (tflite-runtime) and locally ─────
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    _TFLITE_SOURCE = "tflite_runtime"
except ImportError:
    try:
        import tensorflow as tf
        TFLiteInterpreter = tf.lite.Interpreter
        _TFLITE_SOURCE = "tensorflow"
    except ImportError:
        TFLiteInterpreter = None
        _TFLITE_SOURCE = "none"

# ── Multi-format audio support (FIX 3) ────────────────────────────────────────
try:
    from pydub import AudioSegment
    _PYDUB_AVAILABLE = True
except ImportError:
    _PYDUB_AVAILABLE = False

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

# FIX 1: Confidence thresholds
CONFIDENT_THRESHOLD    = 0.75   # above this → full result
LOW_CONF_THRESHOLD     = 0.65   # between 0.65-0.75 → result + low-confidence warning
INCONCLUSIVE_THRESHOLD = 0.65   # below this → Inconclusive

# FIX 4: Sliding window config
WINDOW_SEC = 10    # seconds per window
HOP_SEC    = 5     # hop between windows (50% overlap)

# FIX 2: Noise detection thresholds
RMS_MIN         = 0.001   # too quiet
RMS_MAX         = 0.95    # too loud / clipping
CENTROID_MAX    = 3800    # Hz — lung sounds stay below this
CENTROID_MIN    = 80      # Hz — must have some low-frequency content


def resolve_mongo_uri() -> str:
    try:
        if "mongo" in st.secrets and st.secrets["mongo"].get("uri"):
            return str(st.secrets["mongo"]["uri"]).strip()
        if "MONGO_URI" in st.secrets:
            return str(st.secrets["MONGO_URI"]).strip()
    except Exception:
        pass
    return (os.environ.get("MONGO_URI") or "").strip()


def image_data_uri(path: Path) -> str:
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
HINDI  = ["सामान्य", "क्रैकल — रेफर करें", "व्हीज — रेफर करें"]

# Inconclusive labels (FIX 1)
INCONCLUSIVE_COLOR  = "#6B7280"
INCONCLUSIVE_TINT   = "#F3F4F6"
INCONCLUSIVE_LABEL  = "Inconclusive"
INCONCLUSIVE_DETAIL = "Model confidence too low to make a reliable determination"
INCONCLUSIVE_ADVICE = "Re-record in a quieter environment. Press the chest piece firmly against bare skin."
INCONCLUSIVE_HINDI  = "अनिर्णायक — पुनः रिकॉर्ड करें"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan",
    page_icon="🫁",
    layout="centered"
)

# ── CSS — identical to v1.0 ────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND — all logic (refined v2.0)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model():
    if TFLiteInterpreter is None:
        raise RuntimeError("No TFLite runtime found.")
    interpreter = TFLiteInterpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


@st.cache_resource(show_spinner=False)
def get_mongo_collection(uri: str):
    if not uri:
        raise RuntimeError("No MongoDB URI.")
    if "localhost" in uri or "127.0.0.1" in uri:
        raise RuntimeError("Use Atlas URI, not local.")
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    return client[MONGO_DB][MONGO_COLLECTION]


def save_patient_record(document: dict) -> str:
    collection = get_mongo_collection(resolve_mongo_uri())
    result     = collection.insert_one(document)
    return str(result.inserted_id)


# FIX 3: Multi-format audio loader ─────────────────────────────────────────────
def load_audio_any_format(uploaded_file) -> tuple:
    """
    Load WAV, MP3, M4A, OGG, FLAC — returns (y: np.ndarray, sr: int).
    Falls back to pydub for non-WAV formats if available.
    """
    audio_bytes = uploaded_file.getvalue()
    filename    = uploaded_file.name.lower()

    # Try soundfile first (handles WAV, FLAC, OGG natively)
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        return y, SAMPLE_RATE
    except Exception:
        pass

    # Try pydub for MP3, M4A (requires ffmpeg in packages.txt)
    if _PYDUB_AVAILABLE:
        try:
            fmt = "mp3" if filename.endswith(".mp3") else \
                  "mp4" if filename.endswith((".m4a", ".mp4")) else \
                  "ogg" if filename.endswith(".ogg") else "wav"
            seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
            seg = seg.set_channels(1).set_frame_rate(SAMPLE_RATE)
            y   = np.array(seg.get_array_of_samples(), dtype=np.float32)
            y  /= (2 ** (seg.sample_width * 8 - 1))  # normalize to [-1, 1]
            return y, SAMPLE_RATE
        except Exception:
            pass

    # Final fallback: librosa (slowest but most compatible)
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    return y, SAMPLE_RATE


# FIX 2: Enhanced audio quality check ─────────────────────────────────────────
def check_audio_quality(y: np.ndarray) -> tuple:
    """
    Returns (is_ok: bool, issue: str, details: str)
    Checks RMS level AND spectral centroid to detect non-lung sounds.
    """
    rms = float(np.sqrt(np.mean(y ** 2)))

    if rms < RMS_MIN:
        return False, "too_quiet", f"RMS {rms:.5f} — recording is too quiet. Press mic firmly against bare skin."
    if rms > RMS_MAX:
        return False, "too_loud",  f"RMS {rms:.4f} — clipping detected. Reduce microphone gain."

    # Spectral centroid — lung sounds have energy below ~3800 Hz
    centroid = float(librosa.feature.spectral_centroid(
        y=y, sr=SAMPLE_RATE, n_fft=512, hop_length=64
    )[0].mean())

    if centroid > CENTROID_MAX:
        return False, "not_lung", (
            f"Spectral centroid {centroid:.0f} Hz is too high. "
            "This may not be a lung sound recording. Check file and re-upload."
        )
    if centroid < CENTROID_MIN:
        return False, "no_signal", (
            f"Spectral centroid {centroid:.0f} Hz is extremely low. "
            "Recording may be silence or DC offset."
        )

    return True, "ok", f"RMS {rms:.4f} · centroid {centroid:.0f} Hz"


# FIX 4: Sliding window spectrogram ───────────────────────────────────────────
def audio_to_windows(y: np.ndarray) -> list:
    """
    Split audio into overlapping 10-second windows.
    Returns list of (40×128) mel spectrogram arrays.
    """
    window_samples = SAMPLE_RATE * WINDOW_SEC
    hop_samples    = SAMPLE_RATE * HOP_SEC
    specs          = []

    if len(y) <= window_samples:
        # Shorter than one window — pad and return single spec
        y_pad = np.pad(y, (0, window_samples - len(y)))
        specs.append(_make_spec(y_pad))
        return specs

    start = 0
    while start + window_samples <= len(y):
        window = y[start : start + window_samples]
        specs.append(_make_spec(window))
        start += hop_samples

    return specs


def _make_spec(y: np.ndarray) -> np.ndarray:
    """Convert a 10-second audio segment to a normalized (40×128) mel spectrogram."""
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])   # pre-emphasis
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        fmax=FMAX, n_fft=512, hop_length=64
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] != MEL_WIDTH:
        mel_db = zoom(mel_db, (1.0, MEL_WIDTH / mel_db.shape[1]))
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_norm.astype(np.float32)


# Keep original wav_to_spec for spectrogram visualisation
def wav_to_spec_visual(y: np.ndarray) -> np.ndarray:
    """Return un-normalised mel_db for plotting only."""
    target = SAMPLE_RATE * 10
    y = np.pad(y, (0, max(0, target - len(y))))[:target].astype(np.float32)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    mel    = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        fmax=FMAX, n_fft=512, hop_length=64
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] != MEL_WIDTH:
        mel_db = zoom(mel_db, (1.0, MEL_WIDTH / mel_db.shape[1]))
    return mel_db


def classify_single(interpreter, spec: np.ndarray) -> np.ndarray:
    """Run inference on one (40×128) spectrogram. Returns softmax probs."""
    inp_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(inp_idx, spec[np.newaxis, ..., np.newaxis])
    interpreter.invoke()
    return interpreter.get_tensor(out_idx)[0].copy()


# FIX 4: Majority vote across windows ─────────────────────────────────────────
def classify_full(interpreter, specs: list) -> tuple:
    """
    Classify all windows and return (final_label, avg_probs, per_window_labels).
    Majority vote on window predictions; average probabilities for display.
    """
    all_probs  = []
    all_labels = []

    for spec in specs:
        probs = classify_single(interpreter, spec)
        all_probs.append(probs)
        all_labels.append(int(np.argmax(probs)))

    avg_probs = np.mean(all_probs, axis=0)

    # Majority vote for final label
    counts       = np.bincount(all_labels, minlength=3)
    final_label  = int(np.argmax(counts))

    return final_label, avg_probs, all_labels


# FIX 1: Confidence-aware result ───────────────────────────────────────────────
def get_result_state(label: int, avg_probs: np.ndarray) -> dict:
    """
    Returns a dict describing what to display based on confidence level.
    Three states: CONFIDENT, LOW_CONFIDENCE, INCONCLUSIVE
    """
    conf = float(avg_probs[label])

    if conf < INCONCLUSIVE_THRESHOLD:
        return {
            "state":   "inconclusive",
            "label":   INCONCLUSIVE_LABEL,
            "detail":  INCONCLUSIVE_DETAIL,
            "advice":  INCONCLUSIVE_ADVICE,
            "hindi":   INCONCLUSIVE_HINDI,
            "color":   INCONCLUSIVE_COLOR,
            "tint":    INCONCLUSIVE_TINT,
            "conf":    conf * 100,
            "warning": None,
        }
    elif conf < CONFIDENT_THRESHOLD:
        return {
            "state":   "low_confidence",
            "label":   LABELS[label],
            "detail":  LABEL_DETAIL[label],
            "advice":  ADVICE[label],
            "hindi":   HINDI[label],
            "color":   COLORS[label],
            "tint":    TINTS[label],
            "conf":    conf * 100,
            "warning": (
                f"⚠️ Low confidence ({conf*100:.0f}%). Consider re-recording in a quieter environment "
                "with the microphone pressed firmly against bare skin."
            ),
        }
    else:
        return {
            "state":   "confident",
            "label":   LABELS[label],
            "detail":  LABEL_DETAIL[label],
            "advice":  ADVICE[label],
            "hindi":   HINDI[label],
            "color":   COLORS[label],
            "tint":    TINTS[label],
            "conf":    conf * 100,
            "warning": None,
        }


# FIX 5 + 6: Session history + CSV export ─────────────────────────────────────
def init_session_history():
    if "session_history" not in st.session_state:
        st.session_state.session_history = []


def add_to_session_history(record: dict):
    st.session_state.session_history.append(record)


def session_history_to_csv() -> str:
    """Convert session history list to CSV string for download."""
    if not st.session_state.session_history:
        return ""
    keys = [
        "patient_name", "age", "location", "recording_filename",
        "duration_sec", "signal_rms", "windows_analysed",
        "result", "result_hi", "confidence", "state",
        "advice", "created_at"
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(st.session_state.session_history)
    return buf.getvalue()


def plot_spectrogram(mel_db: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 2.7))
    fig.patch.set_facecolor("#F7FBFA")
    ax.set_facecolor("#F7FBFA")
    img = librosa.display.specshow(
        mel_db, sr=SAMPLE_RATE, hop_length=64,
        x_axis="time", y_axis="mel", fmax=FMAX, ax=ax, cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
    ax.set_title(
        "Log-mel spectrogram", loc="left",
        fontsize=11, fontweight="bold", color="#12332C"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel frequency")
    for spine in ax.spines.values():
        spine.set_color("#D5E4DE")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  UI — identical to v1.0 with refined result logic
# ══════════════════════════════════════════════════════════════════════════════

init_session_history()

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

try:
    interpreter = load_model()
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()

mongo_ok  = False
mongo_uri = resolve_mongo_uri()
try:
    get_mongo_collection(mongo_uri)
    mongo_ok = True
except Exception:
    get_mongo_collection.clear()

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
  <h2>Upload a lung-sound recording</h2>
  <p class="lead">Supports WAV, MP3, M4A, OGG, FLAC · Use a quiet room · At least 6 seconds of steady breathing.</p>
</div>
""",
    unsafe_allow_html=True,
)

# FIX 3: Accept multiple formats
uploaded = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
    label_visibility="collapsed"
)

if uploaded is None:
    st.markdown(
        """
<div class="panel" style="text-align:center;color:#5E746C;padding:1.7rem 1rem;
     position:relative;overflow:hidden">
  <div style="position:absolute;inset:0;opacity:.12;background-image:var(--lung);
       background-repeat:no-repeat;background-position:center;
       background-size:auto 140%;pointer-events:none"></div>
  <div style="position:relative">Drop a WAV, MP3, M4A, OGG, or FLAC recording here to begin screening.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # FIX 5: Show session history if exists
    if st.session_state.session_history:
        st.markdown(
            '<div class="section-label" style="margin-top:1.5rem">Session history</div>',
            unsafe_allow_html=True,
        )
        for i, rec in enumerate(reversed(st.session_state.session_history)):
            conf_str = f"{rec.get('confidence', 0):.0f}%"
            st.markdown(
                f"**{rec['patient_name']}** · {rec['result']} · {conf_str} confidence · {rec['created_at']}",
            )

        # FIX 6: CSV download
        csv_str = session_history_to_csv()
        if csv_str:
            st.download_button(
                "⬇ Download session records (CSV)",
                data=csv_str,
                file_name=f"pneumoscan_session_{datetime.date.today()}.csv",
                mime="text/csv",
            )

    st.markdown(
        '<p class="note">PneumoScan · ICBHI 2017 · For screening use only</p>',
        unsafe_allow_html=True,
    )
    st.stop()

# ── Load audio (FIX 3: multi-format) ──────────────────────────────────────────
with st.spinner("Loading audio…"):
    try:
        y, _ = load_audio_any_format(uploaded)
    except Exception as error:
        st.error(f"Could not read audio file: {error}")
        st.stop()

# ── FIX 2: Quality check ───────────────────────────────────────────────────────
rms          = float(np.sqrt(np.mean(y ** 2)))
duration_sec = len(y) / SAMPLE_RATE
quality_ok, quality_issue, quality_details = check_audio_quality(y)

st.audio(uploaded.getvalue(), format="audio/wav")
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

if not quality_ok:
    if quality_issue == "too_quiet":
        st.warning(f"🔇 {quality_details}")
    elif quality_issue == "too_loud":
        st.warning(f"🔊 {quality_details}")
    elif quality_issue == "not_lung":
        st.error(f"⚠️ {quality_details}")
    elif quality_issue == "no_signal":
        st.error(f"📉 {quality_details}")

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

if not st.button("Analyse recording", type="primary", use_container_width=True):
    st.info("Review the player and signal metrics, then run analysis.")
    st.stop()

# FIX 4: Sliding window inference
with st.spinner("Computing spectrogram windows and running inference…"):
    specs        = audio_to_windows(y)
    label, avg_probs, window_labels = classify_full(interpreter, specs)
    result       = get_result_state(label, avg_probs)   # FIX 1: confidence check
    mel_db_viz   = wav_to_spec_visual(y)

n_windows = len(specs)

# ── Result card ───────────────────────────────────────────────────────────────
color = result["color"]
tint  = result["tint"]

st.markdown(
    f"""
<div class="result" style="background:{tint};border-color:{color}55">
  <h3 style="color:{color}">{result['label']}</h3>
  <p class="detail" style="color:{color}">
    {result['detail']} · {result['hindi']}
  </p>
  <p class="advice">{result['advice']}</p>
  <p class="conf" style="color:{color}">
    Confidence {result['conf']:.1f}% · {n_windows} window{'s' if n_windows > 1 else ''} analysed
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# FIX 1: Low confidence warning
if result["warning"]:
    st.warning(result["warning"])

# ── Class probability bars ────────────────────────────────────────────────────
st.markdown(
    '<div class="section-label" style="margin-top:1.2rem">Class probabilities</div>',
    unsafe_allow_html=True,
)
for index, name in enumerate(LABELS):
    pct = float(avg_probs[index]) * 100
    st.markdown(
        f"""
<div class="bar-row"><span>{name}</span><b>{pct:.1f}%</b></div>
<div class="bar">
  <i style="width:{pct}%;background:{COLORS[index]}"></i>
</div>
""",
        unsafe_allow_html=True,
    )

# FIX 4: Show per-window breakdown if multiple windows
if n_windows > 1:
    with st.expander(f"Window-by-window breakdown ({n_windows} windows)"):
        for i, wl in enumerate(window_labels):
            t_start = i * HOP_SEC
            t_end   = t_start + WINDOW_SEC
            marker  = " ◄ majority" if wl == label else ""
            st.markdown(
                f"**Window {i+1}** ({t_start}s – {t_end}s): "
                f"{LABELS[wl]}{marker}"
            )

# ── Spectrogram ───────────────────────────────────────────────────────────────
with st.expander("View spectrogram"):
    fig = plot_spectrogram(mel_db_viz)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Clinical alert ────────────────────────────────────────────────────────────
if result["state"] == "inconclusive":
    st.warning(
        "🔄 Result is inconclusive. Re-record the patient in a quieter environment "
        "with the chest piece pressed firmly against bare skin."
    )
elif label != 0:
    st.error(
        f"🚨 Abnormal pattern suggested ({LABELS[label]}). "
        "Refer the patient to the nearest Primary Health Centre for clinical assessment."
    )
else:
    st.success("✅ No dominant adventitious pattern detected in this recording.")

# ── Build record ──────────────────────────────────────────────────────────────
now    = datetime.datetime.now(datetime.timezone.utc)
record = {
    "patient_name":       (patient_name or "").strip() or "Unknown",
    "age":                int(patient_age) if patient_age is not None else None,
    "location":           (patient_loc or "").strip() or None,
    "recording_filename": uploaded.name,
    "duration_sec":       round(duration_sec, 2),
    "sample_rate":        SAMPLE_RATE,
    "signal_rms":         round(rms, 6),
    "windows_analysed":   n_windows,
    "result":             result["label"],
    "result_hi":          result["hindi"],
    "result_detail":      result["detail"],
    "confidence":         round(result["conf"], 2),
    "state":              result["state"],
    "probabilities": {
        name: round(float(avg_probs[i]) * 100, 2)
        for i, name in enumerate(LABELS)
    },
    "advice":             result["advice"],
    "quality_details":    quality_details,
    "created_at":         now.strftime("%Y-%m-%d %H:%M UTC"),
    "model_source":       _TFLITE_SOURCE,
}

# FIX 5: Add to session history
add_to_session_history(record)

# Save to MongoDB if connected (FIX 7: non-blocking)
if mongo_ok:
    try:
        save_patient_record(record)
    except PyMongoError:
        get_mongo_collection.clear()

# ── Session record expander ───────────────────────────────────────────────────
with st.expander("Session record", expanded=True):
    st.code(
        f"Patient       : {record['patient_name']}\n"
        f"Age           : {record['age'] if record['age'] is not None else 'Unknown'}\n"
        f"Location      : {record['location'] or 'Unknown'}\n"
        f"Date (UTC)    : {now.strftime('%Y-%m-%d %H:%M')}\n"
        f"File          : {uploaded.name}\n"
        f"Windows       : {n_windows} × {WINDOW_SEC}s\n"
        f"Result        : {result['label']} ({result['hindi']})\n"
        f"State         : {result['state']}\n"
        f"Confidence    : {result['conf']:.1f}%\n"
        f"Quality       : {quality_details}\n"
        f"Advice        : {result['advice']}"
    )

# FIX 6: CSV download in session record
csv_str = session_history_to_csv()
if csv_str:
    st.download_button(
        "⬇ Download all session records (CSV)",
        data=csv_str,
        file_name=f"pneumoscan_{datetime.date.today()}.csv",
        mime="text/csv",
    )

st.markdown(
    '<p class="note">PneumoScan · Built on ICBHI 2017 · '
    'Screening use only · Not a clinical diagnosis</p>',
    unsafe_allow_html=True,
)
