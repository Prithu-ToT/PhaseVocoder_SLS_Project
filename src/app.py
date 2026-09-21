"""
app.py – Phase Vocoder Studio (Streamlit front end)
=====================================================
Time-stretch and pitch-shift audio through a custom STFT phase vocoder.

Architecture used here:
    - `AudioLoader`        (audio_loader.py)      – file I/O + normalize_audio
    - `STFTProcessor`      (stft_processor.py)     – STFT / ISTFT, windowing
    - `vocoder_processor`  (vocoder_processor.py)  – phase-vocoder pipeline

Dispatch rule (per project owner): the "Pitch-Shift Mode" toggle alone
decides which vocoder path runs — Fast always calls `vocoder_process`,
Accurate always calls `vocoder_accurate_process` — regardless of whether
the semitone shift is 0.
"""

import base64
import html as _html
import io
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from audio_loader import AudioLoader, normalize_audio
from stft_processor import STFTProcessor
from vocoder_processor import vocoder_processor

# Fast mode (vocoder_process) rotates STFT phase directly to shift pitch.
# Past a few semitones this smears energy across neighboring frequency
# bins, so it's capped. Accurate mode resamples then time-stretches back,
# so it tolerates a much wider range.
FAST_MODE_MAX_SEMITONES = 3.0
ACCURATE_MODE_MAX_SEMITONES = 12.0

# Speed factor must satisfy 0.20 < speed_factor < 2.5 (vocoder_processor
# .get_synthesis_hop). Keep the slider comfortably inside that range.
SPEED_MIN, SPEED_MAX = 0.25, 2.4

st.set_page_config(page_title="Phase Vocoder Studio", page_icon="🎚️", layout="wide")

# --------------------------------------------------------------------------
# Light visual polish — this is meant to read as a small product, not a
# lab notebook, so we tone down the emoji/step-by-step "assignment" framing.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .pv-hero {
          padding: 1.1rem 1.4rem;
          border-radius: 14px;
          background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
          color: #f9fafb;
          margin-bottom: 1.3rem;
      }
      .pv-hero h1 {
          margin: 0 0 0.25rem 0;
          font-size: 1.6rem;
          font-weight: 700;
      }
      .pv-hero p {
          margin: 0;
          opacity: 0.85;
          font-size: 0.95rem;
      }
      .pv-section-title {
          font-size: 1.05rem;
          font-weight: 600;
          margin: 1.4rem 0 0.5rem 0;
          color: inherit;
      }
    </style>
    <div class="pv-hero">
      <h1>Phase Vocoder Studio</h1>
      <p>Change the speed of a recording without changing its pitch — or shift its pitch without changing its speed — using a custom short-time Fourier phase vocoder.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar — engine internals only
# --------------------------------------------------------------------------
st.sidebar.header("Engine Settings")

frame_size = st.sidebar.select_slider(
    "Frame Size (FFT window)",
    options=[512, 1024, 2048, 4096, 8192],
    value=2048,
    help="Analysis window length. Larger = better frequency resolution; smaller = better time resolution.",
)

overlap_ratio = st.sidebar.slider(
    "Overlap (%)",
    min_value=50,
    max_value=90,
    value=75,
    step=5,
    help="Overlap between successive analysis frames. 75% is standard for Hanning-window reconstruction.",
)

hop_size = int(frame_size * (1 - (overlap_ratio / 100.0)))
st.sidebar.caption(f"Analysis hop size: **{hop_size} samples**")

# --------------------------------------------------------------------------
# Main — 1. Audio source
# --------------------------------------------------------------------------
st.markdown('<div class="pv-section-title">Your Audio</div>', unsafe_allow_html=True)

source_option = st.radio(
    "Audio source",
    options=["Sample track", "Upload a file", "Record with microphone"],
    horizontal=True,
    label_visibility="collapsed",
)

audio_path = None

if source_option == "Sample track":
    default_path = os.path.join("target_io", "mohiner_ghoraguli_sample.mp3")
    if os.path.exists(default_path):
        audio_path = default_path
        st.caption("Using the built-in sample track.")
    else:
        st.error(f"Sample file not found at {default_path}. Please upload or record instead.")

elif source_option == "Upload a file":
    uploaded_file = st.file_uploader(
        "Upload audio", type=["mp3", "wav", "flac", "ogg"], label_visibility="collapsed"
    )
    if uploaded_file is not None:
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        audio_path = temp_file_path
        st.caption(f"Loaded: {uploaded_file.name}")

else:  # Record with microphone
    recorded = st.audio_input("Record audio")
    if recorded is not None:
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, "pv_recorded_input.wav")
        with open(temp_file_path, "wb") as f:
            f.write(recorded.getvalue())
        audio_path = temp_file_path
        st.caption("Recording captured.")

# --------------------------------------------------------------------------
# Main — 2. Vocoder parameters (pulled up from the sidebar on purpose —
# these are the knobs the audience actually cares about)
# --------------------------------------------------------------------------
st.markdown('<div class="pv-section-title">Vocoder Parameters</div>', unsafe_allow_html=True)

param_col1, param_col2 = st.columns(2)

with param_col1:
    speed_factor = st.slider(
        "Speed factor",
        min_value=SPEED_MIN,
        max_value=SPEED_MAX,
        value=1.0,
        step=0.05,
        help="1.0 = original speed. Above 1 speeds up (shortens); below 1 slows down (lengthens).",
    )

with param_col2:
    mode = st.radio(
        "Pitch-shift mode",
        options=["Fast (phase manipulation)", "Accurate (resample + stretch)"],
        help=(
            "Fast rotates STFT phase directly — cheap, but large shifts smear energy across "
            "frequency bins, so it's capped at ±3 semitones. Accurate resamples the signal to "
            "shift pitch, then time-stretches it back with the phase vocoder — higher quality "
            "across a wider range."
        ),
    )

is_fast = mode.startswith("Fast")
max_semitones = FAST_MODE_MAX_SEMITONES if is_fast else ACCURATE_MODE_MAX_SEMITONES

# Keep any previously-chosen shift within the current mode's allowed range
# (switching Accurate -> Fast could otherwise leave a stale out-of-range value).
if "semitone_shift" not in st.session_state:
    st.session_state.semitone_shift = 0.0
st.session_state.semitone_shift = max(
    -max_semitones, min(max_semitones, st.session_state.semitone_shift)
)

semitone_shift = st.slider(
    "Pitch shift (semitones)",
    min_value=-max_semitones,
    max_value=max_semitones,
    step=0.5,
    key="semitone_shift",
)

if is_fast:
    st.caption(
        f"Fast mode is capped at ±{FAST_MODE_MAX_SEMITONES:.0f} semitones — larger shifts smear "
        "energy across STFT bins. Switch to Accurate for bigger shifts."
    )

show_naive = st.checkbox(
    "Also produce a naive-resample comparison clip",
    help=(
        "Resamples the original audio to the same duration change as above using plain "
        "interpolation — no phase vocoder involved. This changes pitch along with speed, "
        "unlike the vocoder output above it, which is the whole point of the demo. The "
        "pitch-shift setting has no effect on this clip."
    ),
)

process_clicked = st.button("Process Audio", type="primary", disabled=audio_path is None)

# --------------------------------------------------------------------------
# Shared plotting helpers — waveform + spectrogram cards, stacked, scrollable,
# on a common horizontal (time) scale with fixed-interval timestamp markers.
# --------------------------------------------------------------------------
PX_PER_SECOND = 120     # shared time scale so waveform & spectrogram widths line up
MIN_PLOT_WIDTH = 900
MAX_PLOT_WIDTH = 12000


def _plot_width(duration: float) -> int:
    return int(max(MIN_PLOT_WIDTH, min(MAX_PLOT_WIDTH, duration * PX_PER_SECOND)))


def _tick_step(duration: float) -> float:
    """Timestamp marker spacing: 10s for shorter clips, 15s otherwise."""
    return 10.0 if duration <= 90 else 15.0


def render_legend(items: list[tuple[str, str]]) -> None:
    swatches = "".join(
        f'<span style="display:inline-flex; align-items:center; margin-right:18px;">'
        f'<span style="width:12px; height:12px; border-radius:3px; background:{color}; '
        f'display:inline-block; margin-right:6px;"></span>'
        f'<span style="font-size:0.85rem; color:#374151;">{_html.escape(label)}</span></span>'
        for label, color in items
    )
    st.markdown(f'<div style="margin:0.1rem 0 0.9rem 0;">{swatches}</div>', unsafe_allow_html=True)


def _waveform_svg(samples: np.ndarray, sample_rate: int, color: str) -> tuple[str, int]:
    """Build a horizontally-scrollable SVG waveform. Returns (svg_markup, width)."""
    n = len(samples)
    height = 160
    mid = height / 2

    if n == 0 or sample_rate <= 0:
        return f'<svg width="900" height="{height}"></svg>', 900

    duration = n / sample_rate
    width = _plot_width(duration)

    target_points = 4000
    step = max(1, n // target_points)
    pts = samples[::step].astype(np.float64)

    peak = max(1e-6, float(np.max(np.abs(pts))))
    xs = np.linspace(0, width, num=len(pts))
    ys = mid - (pts / peak) * (mid * 0.88)
    poly_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    tick_step = _tick_step(duration)
    grid_lines = []
    labels = []
    t = 0.0
    while t <= duration + 1e-9:
        x = (t / duration) * width if duration > 0 else 0.0
        grid_lines.append(
            f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height - 16}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        labels.append(
            f'<text x="{x:.1f}" y="{height - 4}" font-size="10" fill="#9ca3af" '
            f'text-anchor="middle">{int(round(t))}s</text>'
        )
        t += tick_step

    svg = f"""
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
      {''.join(grid_lines)}
      <line x1="0" y1="{mid}" x2="{width}" y2="{mid}" stroke="#d1d5db" stroke-width="1" stroke-dasharray="4,4"/>
      <polyline points="{poly_points}" fill="none" stroke="{color}" stroke-width="1.4"/>
      {''.join(labels)}
    </svg>
    """
    return svg, width


def render_waveform_card(title: str, samples: np.ndarray, sample_rate: int, color: str) -> None:
    svg, width = _waveform_svg(samples, sample_rate, color)
    block = f"""
    <div style="font-family: -apple-system, Segoe UI, Roboto, sans-serif;">
      <div style="font-weight:600; font-size:0.92rem; margin-bottom:6px; color:#374151;">
        {_html.escape(title)}
      </div>
      <div style="overflow-x:auto; overflow-y:hidden; border:1px solid #e5e7eb;
                  border-radius:10px; background:#ffffff; padding:8px;
                  box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
        {svg}
      </div>
    </div>
    """
    components.html(block, height=200, scrolling=False)


def render_spectrogram_card(
    title: str, samples: np.ndarray, sample_rate: int, spec_frame_size: int, spec_hop_size: int
) -> None:
    """Magnitude spectrogram (dB) of `samples`, computed fresh at the
    sidebar's current frame/hop settings, rendered wide + horizontally
    scrollable on the same time scale as the waveform cards above."""
    if len(samples) < spec_frame_size:
        st.caption(f"{title}: clip too short to compute a spectrogram at this frame size.")
        return

    spec_processor = STFTProcessor(frame_size=spec_frame_size, hop_size=spec_hop_size)
    stft_matrix = spec_processor.stft(samples)
    mag_db = 20 * np.log10(np.abs(stft_matrix) + 1e-10)

    duration = len(samples) / sample_rate
    width_px = _plot_width(duration)
    height_px = 260
    dpi = 100

    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax.imshow(
        mag_db,
        origin="lower",
        aspect="auto",
        extent=[0, stft_matrix.shape[1] * spec_hop_size / sample_rate, 0, sample_rate / 2],
        cmap="magma",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    tick_step = _tick_step(duration)
    ticks = np.arange(0, duration + 1e-9, tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(round(t))}" for t in ticks])
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")

    block = f"""
    <div style="font-family: -apple-system, Segoe UI, Roboto, sans-serif;">
      <div style="font-weight:600; font-size:0.92rem; margin-bottom:6px; color:#374151;">
        {_html.escape(title)}
      </div>
      <div style="overflow-x:auto; overflow-y:hidden; border:1px solid #e5e7eb;
                  border-radius:10px; background:#ffffff; padding:8px;
                  box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
        <img src="data:image/png;base64,{b64}" style="display:block; width:{width_px}px; height:{height_px}px;" />
      </div>
    </div>
    """
    components.html(block, height=height_px + 60, scrolling=False)


# --------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------
if process_clicked and audio_path:
    with st.spinner("Running the phase vocoder..."):
        try:
            loader = AudioLoader(audio_path)
            original_audio = loader.audio_data
            sr = loader.sample_rate

            stft_processor = STFTProcessor(frame_size=frame_size, hop_size=hop_size)
            vp = vocoder_processor(loader, stft_processor)

            # Dispatch purely on the selected mode, regardless of semitone value.
            if is_fast:
                pitch_factor = 2.0 ** (semitone_shift / 12.0)
                reconstructed = vp.vocoder_process(speed_factor, pitch_factor=pitch_factor)
            else:
                reconstructed = vp.vocoder_accurate_process(speed_factor, semitone_shift)

            output = normalize_audio(reconstructed)

            naive_output = None
            if show_naive:
                # Plain resample to the same duration change — no phase
                # correction, so pitch shifts along with speed. This is the
                # contrast case: same duration change, no pitch preservation.
                duration_factor = 1.0 / speed_factor
                naive_loader = vp.get_resampled_audioloader(duration_factor)
                naive_output = normalize_audio(naive_loader.audio_data)

            st.markdown('<div class="pv-section-title">Results</div>', unsafe_allow_html=True)

            metric_cols = st.columns(4)
            metric_cols[0].metric("Sample Rate", f"{sr} Hz")
            metric_cols[1].metric("Original Duration", f"{loader.duration_seconds:.2f} s")
            metric_cols[2].metric("Output Duration", f"{len(output) / sr:.2f} s")
            metric_cols[3].metric("Pitch Shift", f"{semitone_shift:+.1f} st", delta=mode.split(" ")[0])

            # Identity sanity check — only meaningful when nothing was changed.
            if speed_factor == 1.0 and semitone_shift == 0.0:
                n = min(len(original_audio), len(reconstructed))
                error = original_audio[:n].astype(np.float64) - reconstructed[:n].astype(np.float64)
                signal_power = np.mean(original_audio[:n].astype(np.float64) ** 2)
                noise_power = np.mean(error ** 2)
                if noise_power > 0:
                    snr_str = f"{10.0 * np.log10(signal_power / noise_power):.1f} dB"
                else:
                    snr_str = "∞ dB (bit-exact)"
                st.caption(f"Identity check (speed 1.0, shift 0 st) — reconstruction SNR: {snr_str}")

            st.markdown("**Waveforms**")
            legend_items = [("Original", "#2563eb"), (f"Processed — {mode}", "#059669")]
            if naive_output is not None:
                legend_items.append(("Naive resample", "#d97706"))
            render_legend(legend_items)

            render_waveform_card("Original", original_audio, sr, "#2563eb")
            render_waveform_card(f"Processed — {mode}", output, sr, "#059669")
            if naive_output is not None:
                render_waveform_card(
                    "Naive resample (pitch shifts too — no phase correction)",
                    naive_output,
                    sr,
                    "#d97706",
                )

            st.markdown("**Spectrograms**")
            render_spectrogram_card("Original — spectrogram", original_audio, sr, frame_size, hop_size)
            render_spectrogram_card(
                f"Processed — {mode} — spectrogram", output, sr, frame_size, hop_size
            )
            if naive_output is not None:
                render_spectrogram_card(
                    "Naive resample — spectrogram", naive_output, sr, frame_size, hop_size
                )

            st.markdown("**Listen**")
            listen_col1, listen_col2 = st.columns(2)
            with listen_col1:
                st.write("Original")
                st.audio(original_audio, format="audio/wav", sample_rate=sr)
            with listen_col2:
                st.write(f"Processed — {mode}")
                st.audio(output, format="audio/wav", sample_rate=sr)

            if naive_output is not None:
                st.write("Naive resample — same duration change, pitch not preserved")
                st.audio(naive_output, format="audio/wav", sample_rate=sr)

        except Exception as e:
            st.error(f"An error occurred during audio processing: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

elif audio_path is None:
    st.info("Choose a sample track, upload a file, or record from your microphone to get started.")