"""
milestone_1.py – Phase Vocoder · Milestone 1
=============================================
STFT Analysis → ISTFT (Overlap-Add) Synthesis → Normalization → Playback.

This module ties together the core building blocks required for the first
milestone of a Phase Vocoder pipeline:

    1. load_and_prep_audio   – Read any common audio format, force mono float32.
    2. create_hanning_window – Build a symmetric Hanning window for framing.
    3. perform_stft          – Short-Time Fourier Transform (analysis).
    4. perform_istft         – Inverse STFT via overlap-add (synthesis).
    5. normalize_audio       – Peak-normalize to [-1.0, 1.0].

No pitch-shifting or time-stretching is performed at this stage.  The round-
trip  load → STFT → ISTFT → normalize → playback  must be *perceptually
transparent* (perfect reconstruction within floating-point tolerance).

Steps 1-2-3-4 above now live in their own classes rather than as loose
functions in this file:

    - `AudioLoader`   (audio_loader.py)   — loading + matplotlib plotting.
    - `STFTProcessor` (stft_processor.py) — STFT / ISTFT with a cached
                                             Hanning window.

`normalize_audio` (step 5) stays here as a small standalone utility, and
this file's `__main__` block shows the classes wired together end to end.

Dependencies
------------
    numpy       – array math & FFT
    soundfile   – robust cross-format audio I/O (libsndfile + FFmpeg),
                  used inside AudioLoader and for writing the demo output
    matplotlib  – waveform plotting, used inside AudioLoader

Authors : Prithu (2305152) & Rafi (2305153)
"""

from __future__ import annotations

import numpy as np

from audio_loader import AudioLoader
from stft_processor import STFTProcessor


# ---------------------------------------------------------------------------
# 5. Audio Normalization
# ---------------------------------------------------------------------------

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize an audio signal to the range [-1.0, 1.0].

    The function finds the maximum absolute sample value and scales the
    entire waveform so that the loudest peak sits at ±1.0.  A small
    headroom factor (0.99) is applied to prevent floating-point rounding
    from nudging any sample above 1.0 — which would cause hard clipping
    in most playback pipelines and DACs.

    If the signal is completely silent (all zeros), it is returned as-is
    to avoid division by zero.

    Parameters
    ----------
    audio : np.ndarray
        1-D real array of arbitrary amplitude range.

    Returns
    -------
    normalized : np.ndarray
        1-D float32 array scaled so that
        ``max(abs(normalized)) ≈ 0.99``.
    """
    # Find the peak absolute value across the entire signal
    peak: float = float(np.max(np.abs(audio)))

    if peak < 1e-10:
        # Signal is essentially silent — nothing to scale.
        return audio.astype(np.float32)

    # Headroom factor keeps us safely below the ±1.0 clipping boundary.
    headroom: float = 0.99

    normalized: np.ndarray = (audio / peak) * headroom

    return normalized.astype(np.float32)


# ---------------------------------------------------------------------------
# Quick-run demo (standalone execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    # --- Resolve directories -------------------------------------------
    # This file now lives in `src/`, while `target_io/` lives one level up,
    # at the project root — so every path below is anchored off that, not
    # off this file's own directory as before.
    SRC_DIR: str = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT: str = os.path.dirname(SRC_DIR)
    TARGET_IO_DIR: str = os.path.join(PROJECT_ROOT, "target_io")

    # --- Resolve the audio file path ------------------------------------
    if len(sys.argv) > 1:
        # Allow the user to pass an arbitrary file path from the CLI
        audio_path: str = sys.argv[1]
    else:
        # Default: use the sample file shipped with the project
        audio_path = os.path.join(TARGET_IO_DIR, "mohiner_ghoraguli_sample.mp3")

    print(f"[milestone_1] Loading audio: {audio_path}")

    # ---- 1. Load & prep -----------------------------------------------
    loader = AudioLoader(audio_path)
    # `AudioLoader.audio_data` / `.sample_rate` are typed `Optional` because
    # an *empty* loader (constructed with no path) is a real, valid state.
    # Here we just constructed it *with* a path, so `load()` already ran
    # and either populated both fields or raised — this assert just makes
    # that guarantee explicit for the type checker (Pylance/pyright), it
    # isn't doing any real error handling of its own.
    assert loader.audio_data is not None and loader.sample_rate is not None
    sr, audio = loader.sample_rate, loader.audio_data
    print(f"  Sample rate : {sr} Hz")
    print(f"  Duration    : {loader.duration_seconds:.2f} s  ({loader.num_samples} samples)")

    # ---- 2-3-4. STFT analysis + ISTFT synthesis (overlap-add) ----------
    FRAME_SIZE: int = 2048   # ~46 ms at 44.1 kHz — good frequency resolution
    HOP_SIZE: int = FRAME_SIZE // 4  # 75 % overlap — standard for Hanning OLA

    processor = STFTProcessor(frame_size=FRAME_SIZE, hop_size=HOP_SIZE)
    print(f"  Frame size  : {processor.frame_size}")
    print(
        f"  Hop size    : {processor.hop_size}  "
        f"(overlap {100 * (1 - processor.hop_size / processor.frame_size):.0f} %)"
    )

    stft_matrix = processor.stft(audio)
    print(f"  STFT shape  : {stft_matrix.shape}  (freq bins × time frames)")

    reconstructed = processor.istft(stft_matrix, expected_length=len(audio))
    print(f"  Reconstructed length: {len(reconstructed)} samples")

    # ---- 5. Normalize ---------------------------------------------------
    output = normalize_audio(reconstructed)
    print(f"  Output dtype: {output.dtype}, peak: {np.max(np.abs(output)):.4f}")

    # ---- 6. Reconstruction error -----------------------------------------
    # Compute the signal-to-noise ratio between original and reconstruction.
    # For perfect reconstruction the SNR should be extremely high (>200 dB
    # with float64 intermediates).
    error = audio[: len(output)].astype(np.float64) - reconstructed[: len(audio)]
    signal_power = np.mean(audio[: len(output)].astype(np.float64) ** 2)
    noise_power = np.mean(error ** 2)

    if noise_power > 0:
        snr_db = 10.0 * np.log10(signal_power / noise_power)
        print(f"  Reconstruction SNR: {snr_db:.1f} dB")
    else:
        print("  Reconstruction SNR: ∞  (bit-exact)")

    # ---- 7. Write the result to disk -------------------------------------
    os.makedirs(TARGET_IO_DIR, exist_ok=True)
    output_path: str = os.path.join(TARGET_IO_DIR, "milestone_1_output.wav")

    import soundfile as sf

    sf.write(output_path, output, sr)
    print(f"  Output saved: {output_path}")

    # ---- 8. Playback (optional, requires sounddevice) ---------------------
    try:
        import sounddevice as sd

        print("  Playing back reconstructed audio …")
        sd.play(output, samplerate=sr)
        sd.wait()  # Block until playback finishes
        print("  Playback complete.")
    except ImportError:
        print("  [INFO] Install 'sounddevice' (`pip install sounddevice`) for live playback.")
    except Exception as playback_err:
        print(f"  [WARN] Playback failed: {playback_err}")

    print("[milestone_1] Done.")
