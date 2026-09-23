"""
audio_loader.py – Audio ingestion & waveform visualization
=============================================================
Wraps audio file I/O (via `soundfile`) and basic waveform plotting
(via `matplotlib`) behind a single small object so the rest of the
pipeline never has to think about file formats, channel counts, or
plotting boilerplate directly.

Authors : Prithu (2305152) & Rafi (2305153)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from pathlib import Path

if TYPE_CHECKING:
    # Only needed for static type checking (Pylance/pyright). matplotlib
    # itself is imported lazily inside `plot()` so simply importing this
    # module doesn't require a display backend / matplotlib install for
    # headless, plot-free use.
    from matplotlib.axes import Axes


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
    # Find the peak absolute value across the entire signal. max(abs(x)) ==
    # max(x.max(), -x.min()) for any real array, so this gets the same peak
    # as np.abs(audio).max() without allocating a full-size abs() copy.
    peak: float = float(max(audio.max(), -audio.min()))

    if peak < 1e-10:
        # Signal is essentially silent — nothing to scale.
        return audio.astype(np.float32)

    # Headroom factor keeps us safely below the ±1.0 clipping boundary.
    headroom: float = 0.99

    # Fold the divide and the headroom multiply into one scale factor so
    # there's only one elementwise pass over the (possibly large) signal
    # instead of two, and let astype's copy (needed anyway for the dtype
    # change) be the only allocation — copy=False skips it entirely on the
    # rare case `audio` is already float32.
    scale: float = headroom / peak
    normalized: np.ndarray = audio * scale

    return normalized.astype(np.float32, copy=False)


class AudioLoader:
    """Loads an audio file from disk into a mono float32 numpy array.

    The loaded signal and its sample rate are stored on the instance
    (`self.audio_data`, `self.sample_rate`) so the object can be passed
    around and reused — e.g. handed to an `STFTProcessor`, or plotted.

    Parameters
    ----------
    file_path : str, optional
        If given, the file is loaded immediately (equivalent to calling
        `.load(file_path)` right after construction). If omitted, the
        loader starts empty and `.load()` must be called explicitly.

    Attributes
    ----------
    file_path : str | None
        Path of the most recently loaded file.
    sample_rate : int | None
        Native sample rate of the loaded audio, in Hz.
    audio_data : np.ndarray | None
        1-D mono float32 array in the range [-1.0, 1.0].

    Notes
    -----
    `file_path` / `sample_rate` / `audio_data` are `None` until a file is
    successfully loaded — this is a genuinely distinct "nothing loaded
    yet" state, not just a typing formality, so callers that need a
    *guaranteed* loaded instance should either check for `None` (see
    `duration_seconds` below for the pattern) or `assert` right after
    construction if a `file_path` was passed in and the load is known to
    have succeeded (any failure would already have raised inside `load`).
    """

    def __init__(self, file_path: str | None = None):
        self.file_path: str | None = None
        self.sample_rate: int | None = None
        self.audio_data: np.ndarray | None = None

        if file_path is not None:
            self.load(file_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, file_path: str) -> AudioLoader:
        """Load an audio file from *file_path*, converting to mono float32.

        Uses `soundfile` (libsndfile + FFmpeg backend), so common formats
        such as WAV, FLAC, OGG, and MP3 are all supported. Multi-channel
        audio is down-mixed to mono by averaging across channels.

        Parameters
        ----------
        file_path : str
            Absolute or relative path to the audio file.

        Returns
        -------
        self : AudioLoader
            Returns itself so calls can be chained, e.g.
            `loader = AudioLoader().load(path)`.

        Raises
        ------
        FileNotFoundError
            If *file_path* does not point to an existing file.
        RuntimeError
            If soundfile cannot decode the file (unsupported codec, corrupt data).
        """
        # `always_2d=True` guarantees shape (samples, channels) even for
        # mono files, which simplifies the down-mix step below.
        audio_data, sample_rate = sf.read(file_path, dtype="float32", always_2d=True)

        if audio_data.shape[1] > 1:
            # Equal-power down-mix: average all channels.
            audio_data = np.mean(audio_data, axis=1)
        else:
            # Squeeze the redundant channel dimension -> (num_samples,)
            audio_data = audio_data[:, 0]

        # Only assigned once every step above has succeeded, so a
        # successful `load()` call always leaves all three attributes
        # populated together — never a partially-loaded state.
        self.audio_data = np.ascontiguousarray(audio_data, dtype=np.float32)
        self.sample_rate = int(sample_rate)
        self.file_path = file_path

        return self

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def num_samples(self) -> int:
        """Number of samples in the loaded signal (0 if nothing loaded)."""
        return 0 if self.audio_data is None else len(self.audio_data)

    @property
    def duration_seconds(self) -> float:
        """Length of the loaded signal in seconds (0.0 if nothing loaded)."""
        # Bind to locals first: narrowing a local variable after an
        # `is None` check is reliable, whereas narrowing repeated
        # `self.attr` accesses can be invalidated by pyright/Pylance
        # around intervening calls. Same pattern used in `plot()` below.
        audio_data = self.audio_data
        sample_rate = self.sample_rate
        if audio_data is None or not sample_rate:
            return 0.0
        return len(audio_data) / sample_rate

    def unload(self, output_path: str | None = None) -> str:
        """Write the loaded audio data to a WAV file.

        Parameters
        ----------
        output_path : str, optional
            Where to write the WAV file. If omitted, the output file is
            derived from the loaded input file's path: created in the same
            directory, with "_output" appended to its filename.

            Example
            -------
            input:
                "music/song.mp3"

            derived output:
                "music/song_output.wav"

        Returns
        -------
        str
            Path of the generated WAV file.

        Raises
        ------
        RuntimeError
            If no audio has been loaded yet, or if `output_path` is omitted
            and no `file_path` is set on this instance to derive one from.
        """
        audio_data = self.audio_data
        sample_rate = self.sample_rate
        file_path = self.file_path

        if audio_data is None or sample_rate is None:
            raise RuntimeError(
                "No audio loaded yet — call `load(file_path)` "
                "or pass `file_path` to the constructor first."
            )

        if output_path is not None:
            final_output_path = Path(output_path)
        else:
            if file_path is None:
                raise RuntimeError(
                    "No output_path given and no file_path is set on this "
                    "AudioLoader to derive one from."
                )
            input_path = Path(file_path)
            final_output_path = input_path.with_name(
                f"{input_path.stem}_output.wav"
            )

        sf.write(
            final_output_path,
            audio_data,
            sample_rate,
            subtype="PCM_16",
        )

        return str(final_output_path)


    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(
        self,
        ax: Axes | None = None,
        title: str | None = None,
        downsample: int | None = None,
    ) -> Axes:
        """Plot the loaded waveform with matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure/axes pair is created if omitted.
        title : str, optional
            Plot title. Defaults to "Waveform — <file_path>".
        downsample : int, optional
            Plot every Nth sample so long files stay responsive to render.
            Auto-computed (targeting ~5000 plotted points) if omitted.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the waveform was drawn on (handy for further tweaks
            or for arranging multiple `AudioLoader` plots side by side).

        Raises
        ------
        RuntimeError
            If no audio has been loaded yet.
        """
        audio_data = self.audio_data
        sample_rate = self.sample_rate
        if audio_data is None or sample_rate is None:
            raise RuntimeError(
                "No audio loaded yet — call `load(file_path)` "
                "or pass `file_path` to the constructor first."
            )

        # Imported lazily so importing this module doesn't require a
        # display backend / matplotlib install for headless, plot-free use.
        import matplotlib.pyplot as plt

        if downsample is None:
            downsample = max(1, len(audio_data) // 5000)

        samples = audio_data[::downsample]
        times = np.arange(len(samples)) * downsample / sample_rate

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))

        ax.plot(times, samples, linewidth=0.6)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title or f"Waveform — {self.file_path}")
        if len(times):
            ax.set_xlim(0, times[-1])

        return ax

    def __repr__(self) -> str:
        if self.audio_data is None:
            return "AudioLoader(empty)"
        return (
            f"AudioLoader(file_path={self.file_path!r}, "
            f"sample_rate={self.sample_rate}, "
            f"duration={self.duration_seconds:.2f}s)"
        )
