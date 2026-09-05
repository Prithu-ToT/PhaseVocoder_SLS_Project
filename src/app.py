"""
app.py – Phase Vocoder Streamlit UI
====================================
Streamlit dashboard for:
- STFT analysis & synthesis
- Perfect reconstruction verification
- Phase-vocoder time scaling
- Original vs. sped-up waveform visualization
- Audio playback comparison
"""

import os
import tempfile

import numpy as np
import streamlit as st

from audio_loader import AudioLoader
from stft_processor import STFTProcessor
from milestone_1 import normalize_audio
from vocoder_processor import vocoder_processor


# --- Directory setup -------------------------------------------------------

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
TARGET_IO_DIR = os.path.join(PROJECT_ROOT, "target_io")


# --- Page configuration ---------------------------------------------------

st.set_page_config(
    page_title="Phase Vocoder - Milestone 1",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Phase Vocoder UI — Milestone 1")

st.markdown(
    """
    Explore the **STFT analysis, synthesis, and phase-vocoder time-scaling**
    pipeline. Upload a custom audio file or use the built-in sample, adjust
    STFT parameters, and control the playback speed using the phase vocoder.
    """
)


# --- Sidebar Controls ------------------------------------------------------

st.sidebar.header("🎛️ STFT Parameters")

frame_size = st.sidebar.select_slider(
    "Frame Size (FFT Window Size)",
    options=[512, 1024, 2048, 4096, 8192],
    value=2048,
    help=(
        "Size of individual window frames. Higher values improve frequency "
        "resolution; lower values improve time resolution."
    ),
)

overlap_ratio = st.sidebar.slider(
    "Overlap Ratio (%)",
    min_value=50,
    max_value=90,
    value=75,
    step=5,
    help="Overlap between successive frames.",
)

hop_size = int(frame_size * (1 - overlap_ratio / 100.0))

st.sidebar.metric(
    label="Calculated Analysis Hop Size",
    value=f"{hop_size} samples",
)


# --- Phase Vocoder Controls ------------------------------------------------

st.sidebar.header("⚡ Phase Vocoder")

speed_factor = st.sidebar.slider(
    "Speed Up Factor",
    min_value=0.20,
    max_value=2.50,
    value=1.00,
    step=0.05,
    help=(
        "1.00× = original speed. Greater than 1.00× speeds up the audio. "
        "Less than 1.00× slows it down."
    ),
)

if speed_factor > 1.0:
    st.sidebar.success(f"Audio will play at {speed_factor:.2f}× speed.")
elif speed_factor < 1.0:
    st.sidebar.info(f"Audio will play at {speed_factor:.2f}× speed.")
else:
    st.sidebar.info("Original speed selected (1.00×).")


# --- Audio Source Selector -------------------------------------------------

st.subheader("🎵 Step 1: Select Audio Source")

source_option = st.radio(
    "Choose input audio",
    options=[
        "Use Sample Audio (Mohiner Ghoraguli)",
        "Upload custom audio file (.mp3, .wav, .flac)",
    ],
    horizontal=True,
)

audio_path = None
uploaded_file = None

if source_option == "Use Sample Audio (Mohiner Ghoraguli)":

    default_path = os.path.join(
        TARGET_IO_DIR,
        "mohiner_ghoraguli_sample.mp3",
    )

    if os.path.exists(default_path):
        audio_path = default_path
        st.info("Using built-in sample audio file.")
    else:
        st.error(
            f"Sample file not found at {default_path}. "
            "Please upload a file instead."
        )

else:

    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["mp3", "wav", "flac", "ogg"],
    )

    if uploaded_file is not None:

        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(
            temp_dir,
            uploaded_file.name,
        )

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        audio_path = temp_file_path

        st.success(
            f"Successfully uploaded: {uploaded_file.name}"
        )


# --- Core Pipeline Execution -----------------------------------------------

if audio_path:

    st.subheader("⚡ Step 2: Run STFT & Phase Vocoder")

    if st.button("🚀 Process Audio", type="primary"):

        with st.spinner(
            "Processing audio through STFT and phase-vocoder stages..."
        ):

            try:

                # ----------------------------------------------------------
                # 1. Load audio
                # ----------------------------------------------------------

                loader = AudioLoader(audio_path)

                assert (
                    loader.audio_data is not None
                    and loader.sample_rate is not None
                )

                sr = loader.sample_rate
                audio = loader.audio_data

                # ----------------------------------------------------------
                # 2. Create STFT processor
                # ----------------------------------------------------------

                processor = STFTProcessor(
                    frame_size=frame_size,
                    hop_size=hop_size,
                )

                # ----------------------------------------------------------
                # 3. Original STFT + ISTFT reconstruction
                # ----------------------------------------------------------

                stft_matrix = processor.stft(audio)

                reconstructed = processor.istft(
                    stft_matrix,
                    expected_length=len(audio),
                )

                # ----------------------------------------------------------
                # 4. Normalize reconstructed original
                # ----------------------------------------------------------

                reconstructed_output = normalize_audio(reconstructed)

                # ----------------------------------------------------------
                # 5. Reconstruction diagnostics
                # ----------------------------------------------------------

                error = (
                    audio.astype(np.float64)
                    - reconstructed.astype(np.float64)
                )

                signal_power = np.mean(
                    audio.astype(np.float64) ** 2
                )

                noise_power = np.mean(error ** 2)

                if noise_power > 0:

                    snr_db = 10.0 * np.log10(
                        signal_power / noise_power
                    )

                    snr_str = f"{snr_db:.1f} dB"

                else:

                    snr_str = "∞ dB (Bit-exact)"

                # ----------------------------------------------------------
                # 6. Run Phase Vocoder
                # ----------------------------------------------------------

                vocoder = vocoder_processor(
                    audio_loader=loader,
                    stft_processor=processor,
                )

                sped_up_audio = vocoder.vocoder_speedup(
                    speed_factor
                )

                sped_up_audio = normalize_audio(
                    sped_up_audio
                )

                # ----------------------------------------------------------
                # 7. Results
                # ----------------------------------------------------------

                st.markdown(
                    "### 📊 Pipeline Results & Verification"
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Sample Rate",
                    f"{sr} Hz",
                )

                col2.metric(
                    "Original Duration",
                    f"{loader.duration_seconds:.2f} s",
                )

                expected_duration = (
                    loader.duration_seconds / speed_factor
                )

                col3.metric(
                    "Sped-up Duration",
                    f"{expected_duration:.2f} s",
                )

                col4.metric(
                    "Reconstruction SNR",
                    snr_str,
                )

                # ----------------------------------------------------------
                # 8. Waveform visualizations
                # ----------------------------------------------------------

                st.markdown(
                    "### 📈 Waveform Visualizations"
                )

                # Downsample only for plotting so Streamlit does not have
                # to render millions of individual samples.

                original_downsample = max(
                    1,
                    len(audio) // 2000,
                )

                sped_up_downsample = max(
                    1,
                    len(sped_up_audio) // 2000,
                )

                original_ds = audio[
                    ::original_downsample
                ]

                sped_up_ds = sped_up_audio[
                    ::sped_up_downsample
                ]

                # ----------------------------------------------------------
                # Original waveform
                # ----------------------------------------------------------

                wave_col1, wave_col2 = st.columns(2)

                with wave_col1:

                    st.write("**Original Signal**")

                    st.line_chart(
                        {
                            "Original Signal": original_ds
                        }
                    )

                # ----------------------------------------------------------
                # Phase-vocoder waveform
                # ----------------------------------------------------------

                with wave_col2:

                    st.write(
                        f"**Phase-Vocoder Output "
                        f"({speed_factor:.2f}×)**"
                    )

                    st.line_chart(
                        {
                            "Sped-up Signal": sped_up_ds
                        }
                    )

                # ----------------------------------------------------------
                # Reconstruction error
                # ----------------------------------------------------------

                error_ds = (
                    error[::original_downsample] * 1000
                )

                st.write(
                    "**STFT Reconstruction Error "
                    "(×1000 magnified)**"
                )

                st.line_chart(
                    {
                        "Reconstruction Error "
                        "(×1000 magnified)": error_ds
                    }
                )

                # ----------------------------------------------------------
                # 9. Audio playback
                # ----------------------------------------------------------

                st.markdown("### 🔊 Listen & Compare")

                playback_col1, playback_col2 = st.columns(2)

                with playback_col1:

                    st.write("**Original Audio:**")

                    st.audio(
                        audio,
                        format="audio/wav",
                        sample_rate=sr,
                    )

                with playback_col2:

                    st.write(
                        f"**Phase-Vocoder Output "
                        f"({speed_factor:.2f}×):**"
                    )

                    st.audio(
                        sped_up_audio,
                        format="audio/wav",
                        sample_rate=sr,
                    )

                # ----------------------------------------------------------
                # 10. Phase Vocoder information
                # ----------------------------------------------------------

                st.markdown("### 🔬 Phase Vocoder Information")

                synthesis_hop = vocoder.get_synthesis_hop(
                    speed_factor
                )

                info_col1, info_col2, info_col3 = st.columns(3)

                info_col1.metric(
                    "Analysis Hop (Ha)",
                    f"{hop_size} samples",
                )

                info_col2.metric(
                    "Synthesis Hop (Hs)",
                    f"{synthesis_hop} samples",
                )

                info_col3.metric(
                    "Speed Factor",
                    f"{speed_factor:.2f}×",
                )

            except Exception as e:

                st.error(
                    f"An error occurred during audio processing: {str(e)}"
                )

                import traceback

                st.code(
                    traceback.format_exc()
                )

else:

    st.info(
        "Please select or upload an audio file to start processing."
    )