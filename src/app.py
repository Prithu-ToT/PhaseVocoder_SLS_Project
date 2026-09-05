"""
app.py – Phase Vocoder Streamlit UI (Milestone 1)
===================================================
A simple web-based dashboard to upload audio, tweak STFT analysis/synthesis
parameters, visualize waveforms, and play back original vs. reconstructed audio.
"""

import streamlit as st
import numpy as np
import os
import tempfile
from milestone_1 import (
    load_and_prep_audio,
    create_hanning_window,
    perform_stft,
    perform_istft,
    normalize_audio,
)

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
    default_path = os.path.join("target_io", "mohiner_ghoraguli_sample.mp3")
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
                sr, audio = load_and_prep_audio(audio_path)
                
                # 2. Window
                win = create_hanning_window(frame_size)
                
                # 3. STFT
                stft_matrix = perform_stft(audio, frame_size, hop_size, win)
                
                # 4. ISTFT
                reconstructed = perform_istft(
                    stft_matrix, frame_size, hop_size, win, expected_length=len(audio)
                )
                
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
                col2.metric("Duration", f"{len(audio) / sr:.2f} seconds")
                col3.metric("Reconstruction SNR", snr_str)
                
                # Waveform visualization using simple line charts for performance
                # Downsample the data for plotting to avoid freezing the browser on long audio
                st.markdown("### 📈 Waveform Visualizations")
                plot_downsample = max(1, len(audio) // 2000)
                
                waveforms = {
                    "Original Signal": audio[::plot_downsample],
                    "Reconstructed Signal": output[::plot_downsample],
                    "Reconstruction Error (x1000 magnified)": error[::plot_downsample] * 1000
                }
                
                st.line_chart(waveforms)

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
