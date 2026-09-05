"""
stft_processor.py – STFT / ISTFT analysis-synthesis engine
=============================================================
Encapsulates the Short-Time Fourier Transform (analysis) and the Inverse
STFT via Overlap-Add / WOLA (synthesis) behind one configurable object.

`frame_size` and `hop_size` can be set at construction time or changed
at runtime (they're plain properties), and the Hanning window used for
analysis/synthesis is generated lazily and cached — it's only rebuilt
when the requested frame size differs from the one currently cached, so
repeated `stft()` / `istft()` calls at a steady frame size pay the
`cos()` generation cost exactly once.

Authors : Prithu (2305152) & Rafi (2305153)
"""

from __future__ import annotations

import numpy as np


class STFTProcessor:
    """STFT analysis / ISTFT (overlap-add) synthesis with window caching.

    Parameters
    ----------
    frame_size : int, default 2048
        FFT / analysis-synthesis window length, in samples.
    hop_size : int, optional
        Stride between successive frames, in samples. Defaults to
        `frame_size // 4` (75% overlap, standard for Hanning WOLA)
        if not given.

    Attributes
    ----------
    frame_size : int
        Current frame size. Settable at any time.
    hop_size : int
        Current hop size. Settable at any time.
    window : np.ndarray
        The (cached) Hanning window for the current `frame_size`.
    """

    def __init__(self, frame_size: int = 2048, hop_size: int | None = None):
        self._frame_size: int = frame_size
        self._hop_size: int = hop_size if hop_size is not None else frame_size // 4

        # Cache for exactly one window: the last frame size it was built
        # for, plus the window itself. Regenerated only on a size change.
        # (Both start out `None` — genuinely "nothing cached yet", not
        # just a typing formality — which is why `_get_window` below
        # re-checks `is None` explicitly rather than trusting the size
        # comparison alone.)
        self._cached_frame_size: int | None = None
        self._cached_window: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Constants exposed as properties so they can be read/set at runtime
    # ------------------------------------------------------------------
    @property
    def frame_size(self) -> int:
        return self._frame_size

    @frame_size.setter
    def frame_size(self, value: int) -> None:
        self._frame_size = value

    @property
    def hop_size(self) -> int:
        return self._hop_size

    @hop_size.setter
    def hop_size(self, value: int) -> None:
        self._hop_size = value

    @property
    def window(self) -> np.ndarray:
        """The Hanning window for the current `frame_size` (cached)."""
        return self._get_window(self._frame_size)

    # ------------------------------------------------------------------
    # Window generation & caching
    # ------------------------------------------------------------------
    def _get_window(self, frame_size: int) -> np.ndarray:
        """Return the Hanning window for `frame_size`.

        Rebuilds it only if `frame_size` differs from whatever is
        currently cached (or nothing is cached yet); otherwise returns
        the cached array as-is.
        """
        if self._cached_frame_size != frame_size or self._cached_window is None:
            self._cached_window = self._create_hanning_window(frame_size)
            self._cached_frame_size = frame_size

        # `_cached_window` is `np.ndarray | None` on the attribute, but the
        # branch above guarantees it's populated by this point — the
        # assert just makes that guarantee explicit for the type checker
        # so the return below is a plain `np.ndarray`, not `Optional`.
        assert self._cached_window is not None
        return self._cached_window

    @staticmethod
    def _create_hanning_window(frame_size: int) -> np.ndarray:
        """Create a symmetric Hanning (raised-cosine) window.

            w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))     for n = 0, 1, ..., N-1

        where N = `frame_size`. The window tapers smoothly to zero at both
        endpoints, which suppresses spectral leakage caused by the
        implicit rectangular truncation of each analysis frame.

        Implemented purely mathematically (no `np.hanning` /
        `scipy.signal.windows` call).

        Returns
        -------
        window : np.ndarray
            1-D float64 array of shape (frame_size,), values in [0.0, 1.0].
        """
        n = np.arange(frame_size, dtype=np.float64)
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (frame_size - 1)))

    # ------------------------------------------------------------------
    # 1. Analysis
    # ------------------------------------------------------------------
    def stft(
        self,
        audio: np.ndarray,
        frame_size: int | None = None,
        hop_size: int | None = None,
    ) -> np.ndarray:
        """Compute the Short-Time Fourier Transform of a 1-D signal.

        Decomposes the time-domain signal into overlapping, windowed
        frames and FFTs each one. Only the non-redundant half of the
        spectrum is kept (bins 0 .. frame_size//2) since the input is
        real-valued.

        Parameters
        ----------
        audio : np.ndarray
            1-D real array (mono audio signal).
        frame_size, hop_size : int, optional
            Override the instance's stored `frame_size` / `hop_size` for
            just this call, without mutating them. Omit to use
            `self.frame_size` / `self.hop_size`.

        Returns
        -------
        stft_matrix : np.ndarray
            2-D complex128 array of shape (num_freq_bins, num_frames)
            where num_freq_bins = frame_size // 2 + 1.
        """
        frame_size = self._frame_size if frame_size is None else frame_size
        hop_size = self._hop_size if hop_size is None else hop_size
        window = self._get_window(frame_size)

        num_samples = len(audio)
        num_frames = 1 + (num_samples - frame_size) // hop_size
        num_freq_bins = frame_size // 2 + 1

        stft_matrix = np.empty((num_freq_bins, num_frames), dtype=np.complex128)

        for frame_idx in range(num_frames):
            start = frame_idx * hop_size
            frame = audio[start : start + frame_size].astype(np.float64)
            windowed_frame = frame * window
            stft_matrix[:, frame_idx] = np.fft.rfft(windowed_frame)

        return stft_matrix

    # ------------------------------------------------------------------
    # 2. Synthesis
    # ------------------------------------------------------------------
    def istft(
        self,
        stft_matrix: np.ndarray,
        expected_length: int,
        frame_size: int | None = None,
        hop_size: int | None = None,
    ) -> np.ndarray:
        """Reconstruct a time-domain signal via Weighted Overlap-Add (WOLA).

        Each column of `stft_matrix` is inverse-FFT'd, multiplied by the
        synthesis window, and additively overlapped into an output buffer.
        A normalization envelope (sum of squared windows per sample) is
        accumulated in parallel and divided out afterwards, undoing the
        amplitude modulation introduced by windowing.

        Parameters
        ----------
        stft_matrix : np.ndarray
            2-D complex array as produced by `stft()`.
        expected_length : int
            Desired output length in samples; the buffer is trimmed or
            zero-padded to match.
        frame_size, hop_size : int, optional
            Must match what was used for analysis. Override the
            instance's stored values for just this call if given.

        Returns
        -------
        output : np.ndarray
            1-D float64 array of shape (expected_length,).
        """
        frame_size = self._frame_size if frame_size is None else frame_size
        hop_size = self._hop_size if hop_size is None else hop_size
        window = self._get_window(frame_size)

        num_frames = stft_matrix.shape[1]
        ola_length = frame_size + (num_frames - 1) * hop_size

        output = np.zeros(ola_length, dtype=np.float64)
        window_sum = np.zeros(ola_length, dtype=np.float64)
        window_sq = window ** 2

        for frame_idx in range(num_frames):
            start = frame_idx * hop_size
            time_frame = np.fft.irfft(stft_matrix[:, frame_idx], n=frame_size)
            output[start : start + frame_size] += time_frame * window
            window_sum[start : start + frame_size] += window_sq

        # Avoid division by zero where no window energy landed (leading /
        # trailing silence outside the frame coverage).
        safe_mask = window_sum > 1e-8
        output[safe_mask] /= window_sum[safe_mask]

        if len(output) >= expected_length:
            output = output[:expected_length]
        else:
            output = np.pad(output, (0, expected_length - len(output)))

        return output

    def __repr__(self) -> str:
        overlap_pct = 100 * (1 - self._hop_size / self._frame_size)
        return (
            f"STFTProcessor(frame_size={self._frame_size}, "
            f"hop_size={self._hop_size}, overlap={overlap_pct:.0f}%)"
        )
