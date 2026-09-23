
import numpy as np

from audio_loader import AudioLoader


class Resampler:
    """Resample `input_audio` so its duration becomes
    `duration * duration_stretch_factor`, using Lanczos-windowed sinc
    interpolation with a matching anti-aliasing low-pass filter whenever
    the output is shorter than the input (`duration_stretch_factor < 1`).
    """

    def __init__(self, input_audio: AudioLoader, duration_stretch_factor: float) -> None:
        self.src = input_audio
        self.s = duration_stretch_factor

        if self.src.audio_data is None or self.s <= 0:
            raise RuntimeError("Invalid Inputs")

    def calculate_output_sample_nums(self) -> int:
        N = self.src.num_samples
        return int(max(1, N * self.s))

    # return an array where re_sampled[i] = src[x], x can be fractional
    def get_resampled_src_indx(self) -> np.ndarray:
        N = self.calculate_output_sample_nums()
        re_sampled = np.arange(N) / self.s  # s = 2x means [out_idx, src_idx] = [0,0], [1,0.5], [2,1], [2n,n]
        return re_sampled

    """
        We are using the Lanczos interpolation kernel with default a = 8.

        The ideal interpolation kernel is sinc(x) out to infinity; Lanczos
        windows it down to zero at a = 8 (17 taps):

            L_a(x) = sinc(x) * sinc(x/a)

        which in the frequency domain is a rect convolved with another rect
        — closer to an ideal brick-wall filter than a plain truncated sinc.

        When we're COMPRESSING the signal (self.s < 1, i.e. fewer output
        samples than input — decimation), reading it out at the new,
        sparser spacing without pre-filtering aliases high-frequency
        content back into the passband. To prevent that we lower the
        kernel's cutoff to the new effective Nyquist and stretch the kernel
        out to match:

            r = min(1, s)                      (cutoff, relative to Nyquist)
            L_a(x) = r * sinc(r*x) * sinc(r*x / a)

        Note it's `s`, not `1/s` — `s` IS already the compression ratio
        (output_len = input_len * s), so it doubles directly as the cutoff.

        `r < 1` widens the kernel's main lobe by 1/r, so the support
        (number of neighboring samples considered) must grow to match:

            support = ceil(a / r)

        When s >= 1 (stretching/upsampling) no extra filtering is needed,
        so r = 1 and support = a, same as plain Lanczos interpolation.
    """

    def calculate_La(self, x, a: int = 8):
        r = min(1.0, self.s)
        rx = r * x
        return r * np.sinc(rx) * np.sinc(rx / a)

    # def interpolate_src_x(self, x, a:int = 8):
    
        #     """  we want values like src[x], how to get them?
        #          take an weighted average of the whole thingy where weight is a
        #     """
    
        #     x_i = int( np.floor(x) )
    
        #     a = 8
        #     r = min(1.0, self.s)
        #     support = int(np.ceil(a / r))
        #     
        #     # for k in xi-support to xi+support, calculate kernel
        #     x_low = max(0, x_i-support)
        #     x_high = min(self.src.num_samples-1, x_i+support)
        #     x_ara = np.arange(x_low, x_high+1)
    
        #     d_ara = (x - x_ara)
        #     kernel = self.calculate_La(d_ara, a)
        #     kernel_sum = np.sum(kernel)
        #     kernel = kernel/kernel_sum if kernel_sum != 0 else kernel
        #     signal = self.src.audio_data[x_low: x_high+1]
    
        #     return np.sum(signal*kernel)
    
        # def get_resampled_audio(self):
            
        #     return_loader = AudioLoader()
        #     return_loader.sample_rate = self.src.sample_rate
    
        #     idx = self.get_resampled_src_indx()
        #     return_loader.audio_data = np.zeros(len(idx), dtype=np.float32)
        #     for i,x in enumerate(idx):
        #         return_loader.audio_data[i] = self.interpolate_src_x(x)
    
        #     return return_loader

    # ai implemented chunk wise vectorized code for longer audio
    def _interpolate_chunk(self, x: np.ndarray, offsets: np.ndarray, a: int = 8) -> np.ndarray:
        """Interpolate one chunk of fractional source positions `x` into
        output samples, using the neighbor `offsets` (precomputed once by
        the caller, since they don't depend on the chunk).

        Returns an array shaped like the source's channel layout:
        `(len(x),)` for mono, `(len(x), channels)` for stereo.
        """
        x_i = np.floor(x).astype(np.int64)

        # Source indices around each x
        indices = x_i[:, None] + offsets[None, :]

        # Which of those indices actually fall inside the source
        valid = (indices >= 0) & (indices < self.src.num_samples)
        indices = np.clip(indices, 0, self.src.num_samples - 1)

        # Lanczos interpolation weights
        distances = x[:, None] - indices
        kernel = self.calculate_La(distances, a)
        kernel *= valid  # zero out contributions outside the signal

        kernel_sum = np.sum(kernel, axis=1, keepdims=True)
        kernel = np.divide(
            kernel, kernel_sum,
            out=np.zeros_like(kernel),
            where=kernel_sum != 0,
        )

        samples = self.src.audio_data[indices]

        if samples.ndim == 3:
            # Stereo: samples = (chunk, taps, channels)
            return np.sum(samples * kernel[:, :, None], axis=1)
        # Mono: samples = (chunk, taps)
        return np.sum(samples * kernel, axis=1)

    def get_resampled_audio(self, chunk_size: int = 250_000) -> AudioLoader:
        """Build the resampled AudioLoader by iterating over the output in
        chunks and delegating the actual interpolation math for each chunk
        to `_interpolate_chunk`."""
        return_loader = AudioLoader()
        return_loader.sample_rate = self.src.sample_rate

        idx = self.get_resampled_src_indx()
        n = len(idx)

        a = 8
        r = min(1.0, self.s)
        support = int(np.ceil(a / r))
        offsets = np.arange(-support, support + 1)

        if self.src.audio_data.ndim == 1:
            output = np.empty(n, dtype=np.float32)
        else:
            channels = self.src.audio_data.shape[1]
            output = np.empty((n, channels), dtype=np.float32)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            output[start:end] = self._interpolate_chunk(idx[start:end], offsets, a)

        return_loader.audio_data = output
        return return_loader