"""
app.py – Phase Vocoder Streamlit UI (Milestone 1)
===================================================
A simple web-based dashboard to upload audio, tweak STFT analysis/synthesis
parameters, visualize waveforms, and play back original vs. reconstructed audio.
"""

import os
import tempfile

import numpy as np
import streamlit as st

from audio_loader import AudioLoader
from stft_processor import STFTProcessor
from milestone_1 import normalize_audio

# --- Directory setup ---------------------------------------------------
# This file lives in `src/`; sample/output audio lives in `target_io/` at
# the project root (one level up), so every path below is anchored off
# this file's own location rather than assuming a particular cwd.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
TARGET_IO_DIR = os.path.join(PROJECT_ROOT, "target_io")

# Set page layout and aesthetics
st.set_page_config(
    page_title="Phase Vocoder - Milestone 1",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Phase Vocoder UI — Milestone 1")
st.markdown(
    """
    Explore the STFT analysis & synthesis pipeline. 
    Upload a custom audio file or use the built-in sample, adjust frame/hop parameters, 
    and verify the **perfect reconstruction** capability of the overlap-add (OLA) system.
    """
)

# --- Sidebar Controls ------------------------------------------------------
st.sidebar.header("🎛️ Parameters")

frame_size = st.sidebar.select_slider(
    "Frame Size (FFT Window Size)",
    options=[512, 1024, 2048, 4096, 8192],
    value=2048,
    help="Size of individual window frames. Higher values improve frequency resolution; lower values improve time resolution.",
)

overlap_ratio = st.sidebar.slider(
    "Overlap Ratio (%)",
    min_value=50,
    max_value=90,
    value=75,
    step=5,
    help="Overlap between successive frames. 75% overlap is standard for Hanning window reconstruction.",
)

# Compute Hop Size from Overlap Ratio
hop_size = int(frame_size * (1 - (overlap_ratio / 100.0)))
st.sidebar.metric(label="Calculated Hop Size", value=f"{hop_size} samples")

# --- Audio Source Selector --------------------------------------------------
st.subheader("🎵 Step 1: Select Audio Source")

source_option = st.radio(
    "Choose input audio",
    options=["Use Sample Audio (Mohiner Ghoraguli)", "Upload custom audio file (.mp3, .wav, .flac)"],
    horizontal=True
)

audio_path = None
uploaded_file = None

if source_option == "Use Sample Audio (Mohiner Ghoraguli)":
    default_path = os.path.join(TARGET_IO_DIR, "mohiner_ghoraguli_sample.mp3")
    if os.path.exists(default_path):
        audio_path = default_path
        st.info("Using built-in sample audio file.")
    else:
        st.error(f"Sample file not found at {default_path}. Please upload a file instead.")
else:
    uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "flac", "ogg"])
    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        audio_path = temp_file_path
        st.success(f"Successfully uploaded: {uploaded_file.name}")

# --- Core Pipeline Execution ------------------------------------------------
if audio_path:
    st.subheader("⚡ Step 2: Run STFT & OLA Reconstruction")
    
    if st.button("🚀 Process Audio", type="primary"):
        with st.spinner("Processing audio through analysis and synthesis stages..."):
            try:
                # 1. Load and prep
                loader = AudioLoader(audio_path)
                # `audio_path` is only ever a real path here (either the
                # bundled sample or a just-written temp upload), so a
                # successful construction always means `load()` fully
                # populated these fields — this assert just states that
                # for the type checker (Pylance/pyright), it's not doing
                # any real error handling of its own.
                assert loader.audio_data is not None and loader.sample_rate is not None
                sr, audio = loader.sample_rate, loader.audio_data

                # 2-3-4. STFT analysis + ISTFT synthesis (window is built
                # and cached inside the processor)
                processor = STFTProcessor(frame_size=frame_size, hop_size=hop_size)
                stft_matrix = processor.stft(audio)
                reconstructed = processor.istft(stft_matrix, expected_length=len(audio))
                
                # 5. Normalization
                output = normalize_audio(reconstructed)
                
                # 6. Diagnostics & metrics
                error = audio.astype(np.float64) - reconstructed.astype(np.float64)
                signal_power = np.mean(audio.astype(np.float64) ** 2)
                noise_power = np.mean(error ** 2)
                
                if noise_power > 0:
                    snr_db = 10.0 * np.log10(signal_power / noise_power)
                    snr_str = f"{snr_db:.1f} dB"
                else:
                    snr_str = "∞ dB (Bit-exact)"
                
                st.markdown("### 📊 Pipeline Results & Verification")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Sample Rate", f"{sr} Hz")
                col2.metric("Duration", f"{loader.duration_seconds:.2f} seconds")
                col3.metric("Reconstruction SNR", snr_str)
                
                # Waveform visualization: original vs. reconstructed get
                # their own side-by-side charts (rather than one combined
                # chart) so each waveform's shape is easy to read on its
                # own axis/scale. The magnified error trace stays as a
                # separate, full-width diagnostic chart underneath.
                st.markdown("### 📈 Waveform Visualizations")
                plot_downsample = max(1, len(audio) // 2000)

                original_ds = audio[::plot_downsample]
                reconstructed_ds = output[::plot_downsample]
                error_ds = error[::plot_downsample] * 1000

                wave_col1, wave_col2 = st.columns(2)

                with wave_col1:
                    st.write("**Original Signal**")
                    st.line_chart({"Original Signal": original_ds})

                with wave_col2:
                    st.write("**Reconstructed Signal**")
                    st.line_chart({"Reconstructed Signal": reconstructed_ds})

                st.write("**Reconstruction Error (×1000 magnified)**")
                st.line_chart({"Reconstruction Error (x1000 magnified)": error_ds})

                # --- Playback Section --------------------------------------
                st.markdown("### 🔊 Listen & Compare")
                
                playback_col1, playback_col2 = st.columns(2)
                
                with playback_col1:
                    st.write("**Original Audio:**")
                    st.audio(audio, format="audio/wav", sample_rate=sr)
                    
                with playback_col2:
                    st.write("**Reconstructed Audio (Normalized):**")
                    st.audio(output, format="audio/wav", sample_rate=sr)
                    
            except Exception as e:
                st.error(f"An error occurred during audio processing: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
else:
    st.info("Please select or upload an audio file to start processing.")
