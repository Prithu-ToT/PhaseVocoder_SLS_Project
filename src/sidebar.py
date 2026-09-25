"""
sidebar.py – every control of Phase Vocoder Studio, in the sidebar
====================================================================
All controls live here together — audio source, vocoder parameters,
engine settings and the Process / Generate Spectrograms buttons — so the
main area is left for results. render_sidebar() draws them and returns
the current choices as one `Settings` object.
"""

import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from string import Template

import streamlit as st

from panel_common import (
    COLOR_BG,
    COLOR_BORDER,
    COLOR_BORDER_HOVER,
    COLOR_BORDER_LIGHT,
    COLOR_SURFACE,
    COLOR_TEXT,
    COLOR_TEXT_BRIGHT,
    COLOR_TEXT_BRIGHTEST,
    COLOR_TEXT_DIM,
    COLOR_TEXT_FAINT,
)

# Fast mode (vocoder_process) rotates STFT phase directly to shift pitch.
# Past a few semitones this smears energy across neighboring frequency
# bins, so it's capped. Accurate mode resamples then time-stretches back,
# so it tolerates a much wider range.
FAST_MODE_MAX_SEMITONES = 3.0
ACCURATE_MODE_MAX_SEMITONES = 12.0

# Speed factor must satisfy 0.20 < speed_factor < 2.5 (vocoder_processor
# .get_synthesis_hop). Keep the slider comfortably inside that range.
SPEED_MIN, SPEED_MAX = 0.25, 2.4

MODE_FAST = "Fast (phase manipulation)"
MODE_ACCURATE = "Accurate (resample + stretch)"

SAMPLE_TRACK_PATH = os.path.join("target_io", "mohiner_ghoraguli_sample.mp3")

# session_state keys shared with app.py: the last processing result, and
# whether its spectrograms have been requested.
RESULTS_KEY = "pv_results"
SHOW_SPECTROGRAMS_KEY = "show_spectrograms"

# Each settings group is an st.container with a key starting with this, so
# SIDEBAR_CSS can style it as a card (Streamlit adds a `st-key-<key>` class).
SECTION_KEY_PREFIX = "pv_sb_"

# Sidebar styling, matched to the result panels (panel_common.PANEL_CSS) —
# both draw from the same COLOR_* tokens in panel_common.py, so they can't
# drift apart the way two independently-hardcoded palettes would. Also
# compact enough that every control fits a maximized 1080p window (~960 px
# of viewport) without scrolling. Targets Streamlit's data-testid hooks.
SIDEBAR_CSS = Template(r"""
<style>
  /* Sidebar surface: the panels' card color, with their border color. */
  [data-testid="stSidebar"] { background: $bg; border-right: 1px solid $border; }
  /* 360 px with slim outer padding: wide enough for the side-by-side
     control pairs below without their labels wrapping. */
  [data-testid="stSidebar"] { min-width: 360px !important; max-width: 360px !important; }
  [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { gap: 0.75rem; }
  [data-testid="stSidebar"] [data-testid="stRadioOption"] { margin-right: 0.9rem; }
  /* The header strip only holds the collapse button — keep it short. */
  [data-testid="stSidebarHeader"] { padding: 0.5rem 0.75rem 0 0.75rem; height: 2.25rem; }
  [data-testid="stSidebarContent"] { padding-left: 0 !important; padding-right: 0 !important; }
  [data-testid="stSidebarUserContent"] { padding: 0 0.6rem 0.75rem 0.6rem !important; }
  /* Upload box: button and file-type hint on one compact row. */
  [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    padding: 0.4rem 0.6rem; flex-direction: row; align-items: center; gap: 0.6rem;
  }
  [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { font-size: 0.72rem; }
  [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] { gap: 0.45rem; }

  /* Settings groups: the panels' control-bar card. */
  [data-testid="stSidebar"] [class*="st-key-pv_sb_"] {
    background: $surface; border: 1px solid $border; border-radius: 10px;
    padding: 8px 10px 10px 10px;
  }
  /* Section title: small grey caps, like the panels' labels. */
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p.pv-sb-title {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; line-height: 1.2;
    text-transform: uppercase; color: $text_dim; margin: 0 0 6px 0;
  }
  /* Sidebar heading, above the settings groups. */
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p.pv-sb-heading {
    font-size: 1.05rem; font-weight: 700; line-height: 1.2;
    color: $text_brightest; margin: 0 0 10px 2px;
  }

  /* Type: the panels use 0.8rem body text, grey labels, dim captions. */
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
  [data-testid="stSidebar"] [data-testid="stRadioOption"] p,
  [data-testid="stSidebar"] [data-testid="stCheckbox"] p {
    font-size: 0.8rem; color: $text;
  }
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] { min-height: 0; margin-bottom: 0; }
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: $text_dim; }
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    font-size: 0.72rem; color: $text_faint; line-height: 1.35;
  }
  [data-testid="stSidebar"] [data-testid="stRadioGroup"] { gap: 0; }
  [data-testid="stSidebar"] [data-testid="stRadioOption"] { min-height: 1.5rem; }

  /* Sliders: the panels' tabular numbers. */
  [data-testid="stSidebar"] [data-testid="stSliderThumbValue"] {
    font-size: 0.72rem; font-variant-numeric: tabular-nums;
  }
  [data-testid="stSidebar"] [data-testid="stSliderTickBar"] {
    font-size: 0.68rem; color: $text_faint; padding-top: 0;
  }

  /* Buttons: the panels' button shape; primary keeps the red accent
     (Streamlit's own theme primaryColor, set in .streamlit/config.toml, so
     the primary "Process Audio" button already matches without an override
     here). */
  [data-testid="stSidebar"] [data-testid="stButton"] button {
    border-radius: 6px; min-height: 2.1rem; padding: 0.25rem 0.75rem; font-size: 0.85rem;
  }
  [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background: $border; border: 1px solid $border_light; color: $text_bright;
  }
  [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover:not(:disabled) {
    background: $border_light; border-color: $border_hover; color: $text_brightest;
  }
  /* Disabled looks disabled, like the panels' buttons (the rules above
     would otherwise make a disabled button look clickable). */
  [data-testid="stSidebar"] [data-testid="stButton"] button:disabled {
    opacity: 0.4; cursor: not-allowed;
  }
</style>
""").substitute(
    bg=COLOR_BG,
    surface=COLOR_SURFACE,
    border=COLOR_BORDER,
    border_light=COLOR_BORDER_LIGHT,
    border_hover=COLOR_BORDER_HOVER,
    text=COLOR_TEXT,
    text_bright=COLOR_TEXT_BRIGHT,
    text_brightest=COLOR_TEXT_BRIGHTEST,
    text_dim=COLOR_TEXT_DIM,
    text_faint=COLOR_TEXT_FAINT,
)


@dataclass(frozen=True)
class Settings:
    """Everything the sidebar controls decide, for one script run."""

    audio_path: str | None  # None until a source is available
    speed_factor: float
    mode: str  # MODE_FAST or MODE_ACCURATE
    semitone_shift: float
    show_naive: bool
    frame_size: int
    hop_size: int
    process_clicked: bool

    @property
    def is_fast(self) -> bool:
        return self.mode == MODE_FAST


def _save_temp(name: str, data: bytes) -> str:
    """Write uploaded/recorded audio to a fresh temp file so AudioLoader can read it.

    Uses a unique path per call rather than the original filename: reusing a
    fixed name (the previous approach) let two uploads/recordings collide on
    the same path — across reruns of one session, across browser tabs, and
    across concurrent users on a shared server — which could hand the reader
    a file mid-write or momentarily locked by another writer.
    """
    suffix = os.path.splitext(name)[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


@contextmanager
def _section(key: str, title: str):
    """A titled settings card (styled by SIDEBAR_CSS)."""
    with st.container(key=SECTION_KEY_PREFIX + key):
        st.markdown(f'<p class="pv-sb-title">{title}</p>', unsafe_allow_html=True)
        yield


def _audio_source() -> str | None:
    """Sample track / upload / microphone picker. Returns a readable path."""

    source_option = st.radio(
        "Audio source",
        options=["Sample track", "Upload a file", "Record with microphone"],
        label_visibility="collapsed",
    )

    if source_option == "Sample track":
        if os.path.exists(SAMPLE_TRACK_PATH):
            st.caption("Using the built-in sample track.")
            return SAMPLE_TRACK_PATH
        st.error(f"Sample file not found at {SAMPLE_TRACK_PATH}. Please upload or record instead.")
        return None

    if source_option == "Upload a file":
        uploaded_file = st.file_uploader(
            "Upload audio", type=["mp3", "wav", "flac", "ogg"], label_visibility="collapsed"
        )
        if uploaded_file is None:
            return None
        path = _save_temp(uploaded_file.name, uploaded_file.getbuffer())
        st.caption(f"Loaded: {uploaded_file.name}")
        return path

    # Record with microphone
    recorded = st.audio_input("Record audio")
    if recorded is None:
        return None
    path = _save_temp("pv_recorded_input.wav", recorded.getvalue())
    st.caption("Recording captured.")
    return path


def render_sidebar() -> Settings:
    """Draw every control in the sidebar and return the current Settings.

    Also records a Generate Spectrograms click in session_state
    (SHOW_SPECTROGRAMS_KEY), so the request survives later reruns.
    """
    with st.sidebar:
        st.html(SIDEBAR_CSS)
        st.markdown('<p class="pv-sb-heading">Control</p>', unsafe_allow_html=True)

        # ---- 1. Audio source ----
        with _section("audio", "Your Audio"):
            audio_path = _audio_source()

        # ---- 2. Vocoder parameters ----
        with _section("vocoder", "Vocoder Parameters"):
            speed_factor, mode, semitone_shift, show_naive = _vocoder_parameters()

        # ---- 3. Engine internals ----
        with _section("engine", "Engine Settings"):
            frame_size, hop_size = _engine_settings()

        # ---- 4. Actions (side by side) ----
        # Wider second column so "Generate Spectrograms" isn't truncated.
        process_col, spectro_col = st.columns([1, 1.35])
        process_clicked = process_col.button(
            "Process Audio", type="primary", disabled=audio_path is None, use_container_width=True
        )

        # Spectrograms are the expensive part of the results, so they're opt-in
        # via their own button. Only usable once a Process run has stored results.
        if spectro_col.button(
            "Generate Spectrograms",
            disabled=RESULTS_KEY not in st.session_state,
            use_container_width=True,
            help="Compute magnitude spectrograms of the processed results at the current engine settings.",
        ):
            st.session_state[SHOW_SPECTROGRAMS_KEY] = True

    return Settings(
        audio_path=audio_path,
        speed_factor=speed_factor,
        mode=mode,
        semitone_shift=semitone_shift,
        show_naive=show_naive,
        frame_size=frame_size,
        hop_size=hop_size,
        process_clicked=process_clicked,
    )


def _vocoder_parameters() -> tuple[float, str, float, bool]:
    """Speed, pitch-shift mode, semitones and the naive-clip checkbox.

    Mode comes first because it sets the pitch slider's range; speed and
    pitch then sit side by side to save height.
    """
    mode = st.radio(
        "Pitch-shift mode",
        options=[MODE_FAST, MODE_ACCURATE],
        help=(
            "Fast rotates STFT phase directly — cheap, but large shifts smear energy across "
            "frequency bins, so it's capped at ±3 semitones. Accurate resamples the signal to "
            "shift pitch, then time-stretches it back with the phase vocoder — higher quality "
            "across a wider range."
        ),
    )

    is_fast = mode == MODE_FAST
    max_semitones = FAST_MODE_MAX_SEMITONES if is_fast else ACCURATE_MODE_MAX_SEMITONES

    # Keep any previously-chosen shift within the current mode's allowed range
    # (switching Accurate -> Fast could otherwise leave a stale out-of-range value).
    if "semitone_shift" not in st.session_state:
        st.session_state.semitone_shift = 0.0
    st.session_state.semitone_shift = max(
        -max_semitones, min(max_semitones, st.session_state.semitone_shift)
    )

    speed_col, pitch_col = st.columns(2)
    speed_factor = speed_col.slider(
        "Speed factor",
        min_value=SPEED_MIN,
        max_value=SPEED_MAX,
        value=1.0,
        step=0.05,
        help="1.0 = original speed. Above 1 speeds up (shortens); below 1 slows down (lengthens).",
    )

    semitone_shift = pitch_col.slider(
        "Pitch shift (semitones)",
        min_value=-max_semitones,
        max_value=max_semitones,
        step=0.5,
        key="semitone_shift",
        help=(
            f"Fast mode is capped at ±{FAST_MODE_MAX_SEMITONES:.0f} semitones — larger shifts "
            "smear energy across STFT bins. Switch to Accurate for up to "
            f"±{ACCURATE_MODE_MAX_SEMITONES:.0f}."
        ),
    )

    if is_fast:
        st.caption(f"Fast mode: capped at ±{FAST_MODE_MAX_SEMITONES:.0f} st — use Accurate for more.")

    show_naive = st.checkbox(
        "Also produce a naive-resample comparison clip",
        help=(
            "Resamples the original audio to the same duration change as above using plain "
            "interpolation — no phase vocoder involved. This changes pitch along with speed, "
            "unlike the vocoder output above it, which is the whole point of the demo. The "
            "pitch-shift setting has no effect on this clip."
        ),
    )
    return speed_factor, mode, semitone_shift, show_naive


def _engine_settings() -> tuple[int, int]:
    """STFT frame size and overlap, side by side. Returns (frame_size, hop_size)."""
    # Wider first column: its label is much longer than "Overlap (%)".
    frame_col, overlap_col = st.columns([1.3, 1])
    frame_size = frame_col.select_slider(
        "Frame Size (FFT window)",
        options=[512, 1024, 2048, 4096, 8192],
        value=2048,
        help="Analysis window length. Larger = better frequency resolution; smaller = better time resolution.",
    )

    overlap_ratio = overlap_col.slider(
        "Overlap (%)",
        min_value=50,
        max_value=90,
        value=75,
        step=5,
        help="Overlap between successive analysis frames. 75% is standard for Hanning-window reconstruction.",
    )

    hop_size = int(frame_size * (1 - (overlap_ratio / 100.0)))
    st.caption(f"Analysis hop size: **{hop_size} samples**")
    return frame_size, hop_size
