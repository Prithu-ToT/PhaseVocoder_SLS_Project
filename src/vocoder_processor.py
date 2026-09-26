"""Phase vocoder pipeline coordination."""

from __future__ import annotations

import numpy as np

from audio_loader import AudioLoader
from stft_processor import STFTProcessor
from lanczos_resampler import Resampler

#added vocoder_note_shift and changed vocoder_speedup to vocoder_process

class vocoder_processor:
    """ Run vocoder processing on audio loaded by an ``AudioLoader``.
        Core Algo is :
            0) synthesis hop Hs = Ha * alpha 
            1) take the stft matrix [bin_indx, frame_indx]
            2) estimate instantenous frequency for each bin
                wk = 2pi * k/N
                wk_hat = wk + delta/Ha

                delta is extra phase advance as the real frquency is often in between bins
                delta = del shi - wk Ha
                delta is zero is frequency perfectly lines up with bin
            
            3) shi_out[first column] = shi[first column]
            shi_out[this column] = shi_out[prev column] + inst_freq* Hs 

            4) vocoder_process mainly processes time, can process fast small pitch shift

        Note: the multi-stage "resample, then stretch back to compensate"
        recipe for accurate pitch shifting (formerly `vocoder_accurate_
        process` on this class) now lives on `VocoderOrchestrator`
        (vocoder_orchestrator.py) — deciding when to resample, when a
        stage can be skipped, and when to persist a result to disk are
        workflow decisions, not phase-vocoder signal processing. This
        class stays focused on STFT / phase-vocoder / ISTFT and on
        `get_resampled_audioloader`, a thin resampling passthrough used
        both by the orchestrator's recipe and by the UI's standalone
        naive-resample comparison. See orchastration.md.
    """

    def __init__(
        self,
        audio_loader: AudioLoader,
        stft_processor: STFTProcessor
    ) -> None:
        self.audio_loader = audio_loader
        self.stft_processor = stft_processor
        self.stft_matrix: np.ndarray | None = None

    def run_stft(self) -> np.ndarray:
        """Compute and store the STFT matrix for the loaded audio."""
        audio_data = self.audio_loader.audio_data
        if audio_data is None:
            raise RuntimeError("No audio loaded in audio_loader.")

        self.stft_matrix = self.stft_processor.stft(audio_data)
        return self.stft_matrix

    def get_synthesis_hop(self, speedup_factor:float)->int:

        if speedup_factor <= 0:
            raise ValueError("Speed factor should be positive")

        if speedup_factor < 0.25 or speedup_factor > 3:
            raise ValueError("Speed factor should be inside 0.25 < sf < 3 range")
        
        return  round(self.stft_processor.hop_size / speedup_factor)


    def calculate_wk_hat(self) -> np.ndarray:

        """
            returns the and angular frequency of all  bins
        """

        if(self.stft_matrix is None):
            self.run_stft()

        if(self.stft_matrix is None):
            raise RuntimeError("run_stft returned none")
                
        shi = np.angle(self.stft_matrix)
        sub_shi = np.roll(a=shi, shift=1, axis=1)
        sub_shi[:,0] = 0
        del_shi = shi - sub_shi

        N = self.stft_processor.frame_size
        Ha = self.stft_processor.hop_size

        #wk for each bin
        k = np.arange(N//2 + 1)[:, None] #2D by shape [ [k0], [k1], ..., [k_N/2]] ready for broadcast to [ [k0, k0....], ...]
        wk = 2*np.pi*k/N

        delta = del_shi - wk*Ha
        #unwrap logic, add +pi to make it non-negative, modulo 2pi makes it bound and then -pi to get back
        # Done in place (delta isn't needed in its pre-wrap form again) so
        # this is 0 extra full-size (bins x frames) allocations instead of 3.
        delta += np.pi
        delta %= (2*np.pi)
        delta -= np.pi
        delta /= Ha
        wk_hat = wk + delta

        return wk_hat

    def compute_spectral_flux(self) -> np.ndarray:
        """
        Per-frame spectral energy flux, used to detect transients (onsets):

            F_t = sum_k max(0, |X_t(k)| - |X_(t-1)(k)|)

        Only the *increase* in magnitude counts (rectified via max(0, ..)) —
        a sudden rise in energy is what marks an attack/transient, a decay
        doesn't need a phase reset.

        Returns
        -------
        flux : np.ndarray
            1-D array of shape (num_frames,). flux[0] is defined as 0.0
            since there is no previous frame to compare frame 0 against.
        """
        if self.stft_matrix is None:
            self.run_stft()
        if self.stft_matrix is None:
            raise RuntimeError("run_stft returned none")

        magnitude = np.abs(self.stft_matrix)
        rise = np.maximum(0.0, magnitude[:, 1:] - magnitude[:, :-1])
        flux = np.sum(rise, axis=0)

        # No previous frame for frame 0 -> flux undefined / can't be a transient.
        return np.concatenate(([0.0], flux))

    def detect_transients(self) -> np.ndarray:
        """
        Flag transient (onset) frames from the spectral flux using an
        adaptive threshold: frame t is a transient if

            F_t > mu_F + 2 * sigma_F

        where mu_F and sigma_F are the mean and standard deviation of the
        flux over the *preceding* ~250ms of frames (frame t itself is
        excluded from its own statistics, otherwise a spike would inflate
        the very threshold it needs to beat).

        A last-transient tracker enforces a ~20ms refractory period after
        each accepted transient so the same attack (whose flux often stays
        elevated for more than one frame) isn't re-triggered on every
        frame it spans.

        Returns
        -------
        is_transient : np.ndarray
            1-D boolean array of shape (num_frames,).
        """
        flux = self.compute_spectral_flux()
        num_frames = len(flux)

        sample_rate = self.audio_loader.sample_rate
        if sample_rate is None:
            raise RuntimeError("Sample rate unavailable.")
        Ha = self.stft_processor.hop_size

        # ~250ms of analysis frames for the running mean/std, ~20ms for the
        # refractory period — both in frame counts (at least 1 frame each,
        # so this still behaves sensibly at very large hop sizes).
        stats_window_frames = max(1, round(0.25 * sample_rate / Ha))
        refractory_frames = max(1, round(0.02 * sample_rate / Ha))

        is_transient = np.zeros(num_frames, dtype=bool)

        # Tracks the frame index of the last accepted transient so a single
        # attack's elevated flux doesn't immediately re-trigger on the next
        # frame(s) within the refractory window.
        last_transient_frame = -refractory_frames

        for t in range(num_frames):
            history = flux[max(0, t - stats_window_frames):t]
            if len(history) < stats_window_frames:
                # Not enough history yet for a stable mean/std (start of the
                # clip) -- with only 1-2 samples, sigma can be ~0 and any
                # nonzero flux would trivially "beat" the threshold.
                continue

            mu = np.mean(history)
            sigma = np.std(history)

            past_refractory = (t - last_transient_frame) >= refractory_frames
            if flux[t] > mu + 2 * sigma and past_refractory:
                is_transient[t] = True
                last_transient_frame = t

        return is_transient

    def calculate_shi_out(self, speed_factor:float = 1, pitch_factor:float = 1):

        Hs = self.get_synthesis_hop(speed_factor)
        wk_hat = self.calculate_wk_hat()
        if pitch_factor != 1.0:
            wk_hat = wk_hat*pitch_factor

        if(self.stft_matrix is None):
            raise RuntimeError("STFT went wrong")

        # Frames flagged as transients get their synthesis phase reset to
        # the analysis phase directly, instead of accumulated from the
        # previous frame. Accumulating straight through an attack smears
        # its energy across the (now differently-spaced) surrounding
        # frames, which is heard as a soft/blurred transient in the output.
        is_transient = self.detect_transients()

        shi_out = np.zeros_like(self.stft_matrix, dtype=float)
        # Only the first column's phase is actually used below, so take the
        # angle of that one column instead of np.angle() over the whole
        # (bins x frames) matrix — calculate_wk_hat() above already did that
        # full-size computation once; no need to redo it here for one column.
        shi_out[:,0] = np.angle(self.stft_matrix[:, 0])
        for frame in range(1, shi_out.shape[1]):
            if is_transient[frame]:
                shi_out[:, frame] = np.angle(self.stft_matrix[:, frame])
            else:
                shi_out[:, frame] = shi_out[:, frame-1] + wk_hat[:, frame]*Hs

        return shi_out

    def vocoder_process(self, speed_factor:float,pitch_factor:float=1.0) -> np.ndarray:
        
        if self.stft_matrix is None:
            self.run_stft()

        shi_out = self.calculate_shi_out(speed_factor,pitch_factor)
        if self.stft_matrix is None:
            raise RuntimeError("Something went wrong")
        stft_out = np.abs(self.stft_matrix) * np.exp(1j*shi_out)
        expected_duration = self.audio_loader.duration_seconds/speed_factor

        if(self.audio_loader.sample_rate is None):
            raise RuntimeError("Something went wrong")
        expected_length = round(expected_duration*self.audio_loader.sample_rate)
        reconstructed_audio = self.stft_processor.istft(stft_out, expected_length, hop_size=self.get_synthesis_hop(speed_factor))
        return reconstructed_audio

    def get_resampled_audioloader(self, duration_factor : float) -> AudioLoader:
        """
            returns an audio with new_duration = duration * duration_factor
        """

        if self.audio_loader.audio_data is None or self.audio_loader.sample_rate is None:
                    raise RuntimeError("No audio loaded in audio_loader.")

        
        resampler = Resampler(self.audio_loader, duration_factor)
        return resampler.get_resampled_audio()