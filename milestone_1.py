"""
milestone_1.py – Phase Vocoder · Milestone 1
=============================================
STFT Analysis → ISTFT (Overlap-Add) Synthesis → Normalization → Playback.

This module provides the five core functions required for the first milestone
of a Phase Vocoder pipeline:

    1. load_and_prep_audio   – Read any common audio format, force mono float32.
    2. create_hanning_window – Build a symmetric Hanning window for framing.
    3. perform_stft          – Short-Time Fourier Transform (analysis).
    4. perform_istft         – Inverse STFT via overlap-add (synthesis).
    5. normalize_audio       – Peak-normalize to [-1.0, 1.0].

No pitch-shifting or time-stretching is performed at this stage.  The round-
trip  load → STFT → ISTFT → normalize → playback  must be *perceptually
transparent* (perfect reconstruction within floating-point tolerance).

Dependencies
------------
    numpy   – array math & FFT
    scipy   – audio file I/O  (scipy.io.wavfile is NOT used; we use
              scipy.io.wavfile's more flexible cousin, soundfile, via
              scipy — but actually we rely on **soundfile** directly for
              broad format support including .mp3, .flac, .ogg, .wav).

Authors : Prithu (2305152) & Rafi (2305153)
"""

from __future__ import annotations

import numpy as np
import soundfile as sf  # Robust cross-format audio I/O (libsndfile + FFmpeg)


# ---------------------------------------------------------------------------
# 1. Audio Loading & Preparation
# ---------------------------------------------------------------------------

def load_and_prep_audio(file_path: str) -> tuple[int, np.ndarray]:
    """Load an audio file from disk, convert to mono float32.

    The function uses the *soundfile* library which delegates to libsndfile
    (and, for compressed formats like MP3, to FFmpeg if available).  The
    returned array is always **1-D float32** regardless of the source format
    (16-bit PCM, 24-bit, stereo, etc.).

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the audio file.  Supported formats
        include WAV, FLAC, OGG, MP3 (if the ``soundfile`` build has FFmpeg
        backend support — most pip installs do).

    Returns
    -------
    sample_rate : int
        Native sample rate of the file in Hz (e.g. 44 100, 48 000).
    audio_data : np.ndarray
        1-D numpy array of shape ``(num_samples,)`` with dtype ``float32``.
        Values are in the range ``[-1.0, 1.0]``.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not point to an existing file.
    RuntimeError
        If soundfile cannot decode the file (unsupported codec, corrupt data).
    """
    # --- Read raw audio; soundfile returns float64 by default ---------------
    # `always_2d=True` guarantees shape (samples, channels) even for mono,
    # which simplifies the stereo→mono conversion below.
    audio_data, sample_rate = sf.read(file_path, dtype="float32", always_2d=True)
    # audio_data shape: (num_samples, num_channels)

    # --- Convert to mono by averaging across channels -----------------------
    # For already-mono files this is a no-op reshape.
    if audio_data.shape[1] > 1:
        # Equal-power down-mix: average all channels.
        audio_data = np.mean(audio_data, axis=1)
    else:
        # Squeeze the redundant channel dimension → (num_samples,)
        audio_data = audio_data[:, 0]

    # --- Ensure contiguous float32 memory layout ----------------------------
    audio_data = np.ascontiguousarray(audio_data, dtype=np.float32)

    return int(sample_rate), audio_data


# ---------------------------------------------------------------------------
# 2. Hanning Window Generation
# ---------------------------------------------------------------------------

def create_hanning_window(frame_size: int) -> np.ndarray:
    """Create a symmetric Hanning (raised-cosine) window.

    The Hanning window is defined as:

        w[n] = 0.5 · (1 − cos(2π·n / (N−1)))     for  n = 0, 1, …, N−1

    where *N* = ``frame_size``.  The window tapers smoothly to **zero** at
    both endpoints, which suppresses *spectral leakage* caused by the
    implicit rectangular truncation of each analysis frame.

    The implementation is purely mathematical (no call to ``np.hanning`` or
    ``scipy.signal.windows``).

    Parameters
    ----------
    frame_size : int
        Number of samples in one analysis/synthesis frame (i.e. the FFT
        size, commonly 1 024, 2 048, or 4 096).

    Returns
    -------
    window : np.ndarray
        1-D float64 array of shape ``(frame_size,)`` containing the window
        coefficients in [0.0, 1.0].
    """
    # Discrete sample indices: 0, 1, 2, …, frame_size − 1
    n = np.arange(frame_size, dtype=np.float64)

    # Raised-cosine formula (symmetric form)
    window: np.ndarray = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (frame_size - 1)))

    return window


# ---------------------------------------------------------------------------
# 3. Short-Time Fourier Transform (Analysis)
# ---------------------------------------------------------------------------

def perform_stft(
    audio: np.ndarray,
    frame_size: int,
    hop_size: int,
    window: np.ndarray,
) -> np.ndarray:
    """Compute the Short-Time Fourier Transform of a 1-D signal.

    The STFT decomposes the time-domain signal into a sequence of
    overlapping, windowed frames and applies the FFT to each one.  The
    result is a 2-D complex matrix whose columns represent successive
    time frames and whose rows represent frequency bins.

    Only the **non-redundant** half of the spectrum is kept (bins
    ``0 … frame_size//2``, i.e. ``frame_size//2 + 1`` bins) because the
    input signal is real-valued.

    Parameters
    ----------
    audio : np.ndarray
        1-D real array (mono audio signal).
    frame_size : int
        Number of samples per analysis frame / FFT size.
    hop_size : int
        Number of samples between successive frame starts (the *stride*).
        A typical choice is ``frame_size // 4`` for 75 % overlap.
    window : np.ndarray
        1-D window array of length ``frame_size`` (e.g. Hanning).

    Returns
    -------
    stft_matrix : np.ndarray
        2-D complex128 array of shape ``(num_freq_bins, num_frames)`` where
        ``num_freq_bins = frame_size // 2 + 1``.  Each column is the FFT of
        one windowed frame.
    """
    # --- Determine the number of complete frames ----------------------------
    num_samples: int = len(audio)
    num_frames: int = 1 + (num_samples - frame_size) // hop_size
    # Only frames that fit entirely within the signal are kept; no zero-
    # padding at the tail (keeps reconstruction simpler and bit-exact).

    # Number of unique frequency bins for a real-valued signal
    num_freq_bins: int = frame_size // 2 + 1

    # --- Pre-allocate the output matrix (frequency × time) ------------------
    stft_matrix: np.ndarray = np.empty(
        (num_freq_bins, num_frames), dtype=np.complex128
    )

    # --- Frame-by-frame analysis loop ---------------------------------------
    for frame_idx in range(num_frames):
        # Starting sample index of this frame
        start: int = frame_idx * hop_size

        # Extract the frame from the audio signal
        frame: np.ndarray = audio[start : start + frame_size].astype(np.float64)

        # Apply the analysis window (element-wise multiplication)
        windowed_frame: np.ndarray = frame * window

        # Compute the FFT; np.fft.rfft returns only non-negative frequencies
        spectrum: np.ndarray = np.fft.rfft(windowed_frame)

        # Store the complex spectrum as one column of the STFT matrix
        stft_matrix[:, frame_idx] = spectrum

    return stft_matrix


# ---------------------------------------------------------------------------
# 4. Inverse STFT (Synthesis via Overlap-Add)
# ---------------------------------------------------------------------------

def perform_istft(
    stft_matrix: np.ndarray,
    frame_size: int,
    hop_size: int,
    window: np.ndarray,
    expected_length: int,
) -> np.ndarray:
    """Reconstruct a time-domain signal from its STFT via overlap-add (OLA).

    Each column of the STFT matrix is inverse-FFT'd, multiplied by the
    *synthesis window*, and additively overlapped into an output buffer.
    A **normalization envelope** (the sum of squared windows at each sample
    position) is accumulated in parallel so that the final signal can be
    divided point-wise to undo the amplitude modulation introduced by
    windowing.

    For a Hanning window with 75 % overlap (``hop_size = frame_size // 4``),
    the squared-window sum is nearly constant, yielding transparent
    reconstruction.  The normalization step handles any residual ripple.

    Parameters
    ----------
    stft_matrix : np.ndarray
        2-D complex array of shape ``(num_freq_bins, num_frames)`` as
        produced by :func:`perform_stft`.
    frame_size : int
        Number of samples per synthesis frame (must match the analysis
        frame size).
    hop_size : int
        Hop size used during analysis.
    window : np.ndarray
        1-D synthesis window of length ``frame_size``.  Using the **same**
        Hanning window as in analysis gives the standard Weighted
        Overlap-Add (WOLA) reconstruction.
    expected_length : int
        The desired length (in samples) of the output signal.  The
        reconstructed buffer is truncated (or, rarely, zero-padded) to
        this length so it matches the original input exactly.

    Returns
    -------
    output : np.ndarray
        1-D float64 array of shape ``(expected_length,)`` containing the
        reconstructed time-domain signal.
    """
    num_frames: int = stft_matrix.shape[1]

    # --- Total length of the OLA buffer (may be slightly longer than the
    #     original signal because the last frame can extend past the end) ----
    ola_length: int = frame_size + (num_frames - 1) * hop_size

    # Accumulator for the reconstructed waveform
    output: np.ndarray = np.zeros(ola_length, dtype=np.float64)

    # Accumulator for the squared-window normalization envelope.
    # This tracks how much "window energy" lands on each sample so we can
    # perfectly undo it after the OLA loop.
    window_sum: np.ndarray = np.zeros(ola_length, dtype=np.float64)

    # Pre-compute the squared window (used for the normalization denominator)
    window_sq: np.ndarray = window ** 2

    # --- Frame-by-frame synthesis loop --------------------------------------
    for frame_idx in range(num_frames):
        # Starting sample of this frame in the output buffer
        start: int = frame_idx * hop_size

        # Inverse FFT: frequency domain → time domain for this frame.
        # np.fft.irfft expects the non-redundant half-spectrum and returns
        # a real array of length `frame_size`.
        time_frame: np.ndarray = np.fft.irfft(
            stft_matrix[:, frame_idx], n=frame_size
        )

        # Apply the synthesis window and overlap-add into the output buffer
        output[start : start + frame_size] += time_frame * window

        # Accumulate the squared window for normalization
        window_sum[start : start + frame_size] += window_sq

    # --- Normalize by the squared-window envelope ---------------------------
    # Avoid division by zero in regions where no window energy landed
    # (leading/trailing silence).  A small epsilon guard is sufficient.
    safe_mask: np.ndarray = window_sum > 1e-8
    output[safe_mask] /= window_sum[safe_mask]

    # --- Trim (or pad) to the expected output length ------------------------
    if len(output) >= expected_length:
        output = output[:expected_length]
    else:
        # Edge case: pad with zeros if OLA buffer is shorter than expected
        output = np.pad(output, (0, expected_length - len(output)))

    return output


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
    import sys
    import os

    # --- Resolve the audio file path ----------------------------------------
    if len(sys.argv) > 1:
        # Allow the user to pass an arbitrary file path from the CLI
        audio_path: str = sys.argv[1]
    else:
        # Default: use the sample file shipped with the project
        script_dir: str = os.path.dirname(os.path.abspath(__file__))
        audio_path = os.path.join(script_dir, "target_io", "mohiner_ghoraguli_sample.mp3")

    print(f"[milestone_1] Loading audio: {audio_path}")

    # ---- 1. Load & prep ----------------------------------------------------
    sr, audio = load_and_prep_audio(audio_path)
    print(f"  Sample rate : {sr} Hz")
    print(f"  Duration    : {len(audio) / sr:.2f} s  ({len(audio)} samples)")

    # ---- 2. Analysis parameters --------------------------------------------
    FRAME_SIZE: int = 2048   # ~46 ms at 44.1 kHz — good frequency resolution
    HOP_SIZE: int   = FRAME_SIZE // 4  # 75 % overlap — standard for Hanning OLA

    win = create_hanning_window(FRAME_SIZE)
    print(f"  Frame size  : {FRAME_SIZE}")
    print(f"  Hop size    : {HOP_SIZE}  (overlap {100 * (1 - HOP_SIZE / FRAME_SIZE):.0f} %)")

    # ---- 3. STFT (analysis) ------------------------------------------------
    stft = perform_stft(audio, FRAME_SIZE, HOP_SIZE, win)
    print(f"  STFT shape  : {stft.shape}  (freq bins × time frames)")

    # ---- 4. ISTFT (synthesis via overlap-add) ------------------------------
    reconstructed = perform_istft(stft, FRAME_SIZE, HOP_SIZE, win, expected_length=len(audio))
    print(f"  Reconstructed length: {len(reconstructed)} samples")

    # ---- 5. Normalize ------------------------------------------------------
    output = normalize_audio(reconstructed)
    print(f"  Output dtype: {output.dtype}, peak: {np.max(np.abs(output)):.4f}")

    # ---- 6. Reconstruction error -------------------------------------------
    # Compute the signal-to-noise ratio between original and reconstruction.
    # For perfect reconstruction the SNR should be extremely high (>200 dB
    # with float64 intermediates).
    error = audio[:len(output)].astype(np.float64) - reconstructed[:len(audio)]
    signal_power = np.mean(audio[:len(output)].astype(np.float64) ** 2)
    noise_power  = np.mean(error ** 2)

    if noise_power > 0:
        snr_db = 10.0 * np.log10(signal_power / noise_power)
        print(f"  Reconstruction SNR: {snr_db:.1f} dB")
    else:
        print("  Reconstruction SNR: ∞  (bit-exact)")

    # ---- 7. Write the result to disk ---------------------------------------
    output_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "target_io",
        "milestone_1_output.wav",
    )
    sf.write(output_path, output, sr)
    print(f"  Output saved: {output_path}")

    # ---- 8. Playback (optional, requires sounddevice) ----------------------
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
