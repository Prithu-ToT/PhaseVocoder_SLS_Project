"""
processing.py – run one Process Audio request
===============================================
Loads the audio, runs the vocoder path the Pitch-shift mode selects, the
optional naive-resample comparison, and the identity SNR check, and returns
everything the results area shows as one `ProcessingResult`. No Streamlit
UI here — errors are raised to the caller.

Dispatch rule (per project owner): the Pitch-shift mode alone decides which
vocoder path runs — Fast always calls `orchestrator.vocoder_process`,
Accurate always calls `orchestrator.process_and_write` — regardless of
whether the semitone shift is 0.
"""

from dataclasses import dataclass

import numpy as np

from audio_loader import AudioLoader, normalize_audio
from stft_processor import STFTProcessor
from vocoder_orchestrator import VocoderOrchestrator


@dataclass
class ProcessingResult:
    """One processed clip set, kept in session_state between reruns."""

    sample_rate: int
    original: np.ndarray
    original_duration: float  # seconds
    output: np.ndarray  # vocoder output, peak-normalized
    naive: np.ndarray | None  # naive-resample comparison clip, if requested
    mode: str  # the Pitch-shift mode label it was processed with
    semitone_shift: float
    snr: str | None  # identity-check SNR, only when speed 1.0 and shift 0


def _identity_snr(original: np.ndarray, reconstructed: np.ndarray) -> str:
    """Reconstruction SNR of an identity run (speed 1.0, shift 0), as text."""
    n = min(len(original), len(reconstructed))
    # Cast once and reuse for both the error and the signal power.
    original_f64 = original[:n].astype(np.float64)
    error = original_f64 - reconstructed[:n].astype(np.float64)
    signal_power = np.mean(original_f64 ** 2)
    noise_power = np.mean(error ** 2)
    if noise_power > 0:
        return f"{10.0 * np.log10(signal_power / noise_power):.1f} dB"
    return "∞ dB (bit-exact)"


def run_processing(
    audio_path: str,
    speed_factor: float,
    semitone_shift: float,
    fast: bool,
    mode_label: str,
    frame_size: int,
    hop_size: int,
    include_naive: bool,
) -> ProcessingResult:
    """Process `audio_path` and return the result.

    `fast` picks the vocoder path (see the dispatch rule above);
    `mode_label` is only recorded on the result for display.
    """
    loader = AudioLoader(audio_path)
    original_audio = loader.audio_data
    sr = loader.sample_rate

    stft_processor = STFTProcessor(frame_size=frame_size, hop_size=hop_size)
    orchestrator = VocoderOrchestrator(loader, stft_processor)

    # Dispatch purely on the selected mode, regardless of semitone value.
    if fast:
        pitch_factor = 2.0 ** (semitone_shift / 12.0)
        reconstructed = orchestrator.vocoder_process(speed_factor, pitch_factor=pitch_factor)
    else:
        # Accurate mode also writes the result to disk (via
        # AudioLoader.unload(), next to the input file) as a side
        # effect — no UI change, this is an internal persistence
        # step of the orchestration workflow.
        reconstructed = orchestrator.process_and_write(speed_factor, semitone_shift)

    output = normalize_audio(reconstructed)

    naive_output = None
    if include_naive:
        # Plain resample to the same duration change — no phase
        # correction, so pitch shifts along with speed. This is the
        # contrast case: same duration change, no pitch preservation.
        duration_factor = 1.0 / speed_factor
        naive_loader = orchestrator.get_resampled_audioloader(duration_factor)
        naive_output = normalize_audio(naive_loader.audio_data)

    # Identity sanity check — only meaningful when nothing was changed.
    snr = None
    if speed_factor == 1.0 and semitone_shift == 0.0:
        snr = _identity_snr(original_audio, reconstructed)

    return ProcessingResult(
        sample_rate=sr,
        original=original_audio,
        original_duration=loader.duration_seconds,
        output=output,
        naive=naive_output,
        mode=mode_label,
        semitone_shift=semitone_shift,
        snr=snr,
    )
