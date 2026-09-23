"""vocoder_orchestrator.py – Application-level workflow coordination
=============================================================
`VocoderOrchestrator` is the API surface `app.py` (or any other caller)
talks to. It answers "which transformations should I run, and in what
order?" — the processing classes (`Resampler`, `STFTProcessor`,
`vocoder_processor`) only answer "how do I perform one transformation?".

This module intentionally contains no new signal-processing math. The
accurate-pitch-shift recipe implemented in `process_and_write()` below is
the same recipe that used to live in `vocoder_processor.vocoder_accurate_
process()` — it has been moved here unchanged, because deciding *when* to
resample, *when* the phase vocoder can be skipped, and *when* to persist
the result to disk are workflow/policy decisions, not phase-vocoder DSP.
See `orchastration.md` for the full rationale.

"""

from __future__ import annotations

import numpy as np

from audio_loader import AudioLoader
from stft_processor import STFTProcessor
from vocoder_processor import vocoder_processor


class VocoderOrchestrator:
    """Coordinates `Resampler`/`vocoder_processor` calls into the
    higher-level operations the UI actually needs, and owns writing
    results to disk.

    Parameters
    ----------
    audio_loader : AudioLoader
        The source audio to process. Must already have `load()`ed a file
        (i.e. `audio_loader.audio_data` / `.sample_rate` are populated).
    stft_processor : STFTProcessor
        STFT/ISTFT engine (frame size, hop size) shared by every
        phase-vocoder stage this orchestrator runs.

    Attributes
    ----------
    audio_loader : AudioLoader
        The source loader passed in at construction.
    stft_processor : STFTProcessor
        The STFT engine passed in at construction.
    processor : vocoder_processor
        The underlying `vocoder_processor` bound to `audio_loader` /
        `stft_processor`, used for the "fast" (direct phase-manipulation)
        path and as the resampling entry point for the "accurate" path.
    """

    def __init__(self, audio_loader: AudioLoader, stft_processor: STFTProcessor) -> None:
        self.audio_loader = audio_loader
        self.stft_processor = stft_processor
        self.processor = vocoder_processor(audio_loader, stft_processor)

    # ------------------------------------------------------------------
    # API endpoint: Fast mode (direct STFT phase manipulation)
    # ------------------------------------------------------------------
    def vocoder_process(self, speed_factor: float, pitch_factor: float = 1.0) -> np.ndarray:
        """Fast pitch/speed change via direct phase manipulation.

        A thin passthrough to `vocoder_processor.vocoder_process` — this
        is a single, genuine signal-processing operation (no multi-stage
        recipe, nothing to coordinate), so there's no orchestration logic
        to add. It's exposed here anyway so `app.py` has one API surface
        (the orchestrator) for every processing path, fast or accurate.

        Parameters
        ----------
        speed_factor : float
            Output duration = input duration / speed_factor.
        pitch_factor : float, default 1.0
            Multiplies the estimated instantaneous frequency directly.
            `app.py` derives this from the semitone slider
            (`2.0 ** (semitone_shift / 12.0)`) before calling in.

        Returns
        -------
        np.ndarray
            The processed audio. Not written to disk — only the accurate
            pipeline (`process_and_write`) persists a result.
        """
        return self.processor.vocoder_process(speed_factor, pitch_factor)

    # ------------------------------------------------------------------
    # API endpoint: naive-resample comparison clip
    # ------------------------------------------------------------------
    def get_resampled_audioloader(self, duration_factor: float) -> AudioLoader:
        """Plain Lanczos-resample comparison clip (no phase correction).

        A thin passthrough to `vocoder_processor.get_resampled_audioloader`.
        This exists purely for the UI's naive-resample demo — it is *not*
        part of the accurate pitch-shift recipe below and deliberately
        ignores pitch entirely, which is the point of the comparison.

        Parameters
        ----------
        duration_factor : float
            New duration = old duration * duration_factor.

        Returns
        -------
        AudioLoader
            A new loader holding the resampled (pitch-shifted-as-a-side-
            effect) audio.
        """
        return self.processor.get_resampled_audioloader(duration_factor)

    # ------------------------------------------------------------------
    # API endpoint: accurate pitch shift (resample + compensating stretch)
    # ------------------------------------------------------------------
    def process_and_write(
        self,
        speed_factor: float,
        semitone_shift: float,
        output_path: str | None = None,
    ) -> np.ndarray:
        """Run the accurate pitch-shift recipe and persist the result.

        This is the multi-stage workflow formerly implemented as
        `vocoder_processor.vocoder_accurate_process` (moved here verbatim,
        math untouched — see `orchastration.md` Sections 5-10 for why it
        belongs in the orchestrator rather than in `vocoder_processor`):

            1. Convert `semitone_shift` -> a resampling ratio
               (`pitch_time_stretch = 2 ** (-semitone_shift / 12.0)`).
            2. Resample the source audio by that ratio (shifts pitch,
               changes duration).
            3. Calculate the phase-vocoder speed factor needed to bring
               the resampled audio's duration back in line with what
               `speed_factor` actually asked for
               (`true_speed_factor = pitch_time_stretch * speed_factor`).
            4. Trivial-case optimizations, preserved unchanged:
               - `semitone_shift == 0` -> skip resampling entirely, run
                 the phase vocoder directly on the original audio.
               - `true_speed_factor ≈ 1` -> the resample step already
                 produced the correct duration; skip the phase vocoder
                 and return the resampled audio as-is.
            5. Otherwise, run the phase vocoder on the resampled audio to
               compensate for the leftover duration change.
            6. Write the final result to disk via `AudioLoader.unload()`
               and return the numpy array.

        Parameters
        ----------
        speed_factor : float
            Desired output duration = input duration / speed_factor.
        semitone_shift : float
            Desired pitch shift in semitones (positive = up).
        output_path : str, optional
            Where to write the resulting WAV file. If omitted, `unload()`
            derives one next to the original input file
            (`<input_stem>_output.wav`) — see `AudioLoader.unload`.

        Returns
        -------
        np.ndarray
            The final processed audio (the same array that was written
            to disk).

        Raises
        ------
        RuntimeError
            If no audio has been loaded into `self.audio_loader`.
        """
        audio_loader = self.audio_loader
        if audio_loader.audio_data is None or audio_loader.sample_rate is None:
            raise RuntimeError("No audio loaded in audio_loader.")

        if semitone_shift == 0:
            # No pitch-changing resample needed at all — run the phase
            # vocoder directly on the original audio. Avoids creating a
            # temporary AudioLoader and running the resampler for nothing.
            final_audio = self.processor.vocoder_process(speed_factor)
        else:
            pitch_time_stretch = 2 ** (-semitone_shift / 12.0)

            # Step 1: resample to shift pitch (changes duration too).
            # Reuses vocoder_processor.get_resampled_audioloader — same
            # Resampler call the original vocoder_accurate_process made —
            # rather than duplicating the Resampler wiring here.
            resampled_audioloader = self.processor.get_resampled_audioloader(
                pitch_time_stretch
            )

            # Step 2: figure out how much time-stretch is still needed to
            # land on the requested speed_factor after that resample.
            true_speed_factor = pitch_time_stretch * speed_factor

            if abs(true_speed_factor - 1) < 1e-5:
                # The resample step already produced the correct duration
                # on its own — skip the phase vocoder entirely.
                final_audio = resampled_audioloader.audio_data
            else:
                # Do NOT unload/discard resampled_audioloader before this
                # runs — its audio_data is exactly what the time-stretch
                # stage below needs (see orchastration.md Section 10).
                time_vocoder = vocoder_processor(resampled_audioloader, self.stft_processor)
                final_audio = time_vocoder.vocoder_process(true_speed_factor, pitch_factor=1.0)

        self._write_result(final_audio, output_path)
        return final_audio

    # ------------------------------------------------------------------
    # Internal: persist a finished result via AudioLoader.unload()
    # ------------------------------------------------------------------
    def _write_result(self, audio_data: np.ndarray, output_path: str | None) -> str:
        """Wrap `audio_data` in a throwaway `AudioLoader` and `unload()`
        it to disk.

        `AudioLoader.unload()` is the one method that knows how to write a
        WAV file (sample rate, subtype, path derivation), so file I/O for
        orchestrator results goes through it rather than calling
        `soundfile` directly here — keeping "how audio gets written" in
        exactly one place.

        `unload()` does not clear `audio_data` on the instance it's called
        on, so the caller-owned `audio_data` array returned by
        `process_and_write` above stays valid after this call.
        """
        result_loader = AudioLoader()
        result_loader.audio_data = audio_data
        result_loader.sample_rate = self.audio_loader.sample_rate
        result_loader.file_path = self.audio_loader.file_path
        return result_loader.unload(output_path)

    def __repr__(self) -> str:
        return (
            f"VocoderOrchestrator(audio_loader={self.audio_loader!r}, "
            f"stft_processor={self.stft_processor!r})"
        )
