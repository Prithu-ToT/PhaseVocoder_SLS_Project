"""Phase vocoder pipeline coordination."""

from __future__ import annotations

import numpy as np

from audio_loader import AudioLoader
from stft_processor import STFTProcessor

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
                
    """

    def __init__(
        self,
        audio_loader: AudioLoader,
        stft_processor: STFTProcessor
    ) -> None:
        self.audio_loader = audio_loader
        self.stft_processor = stft_processor
        self.stft_matrix: np.ndarray | None = None
        self.run_stft()

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

        if speedup_factor < 0.20 or speedup_factor > 2.5:
            raise ValueError("Speed factor should be inside 0.20 < sf < 2.5 range")
        
        return  round(self.stft_processor.hop_size / speedup_factor)


    def calculate_wk_hat(self) -> np.ndarray:

        """
            returns the and angular frequency of all  bins
        """

        if(self.stft_matrix is None):
            raise RuntimeError("Audio was not stft-ed")
        
        shi = np.angle(self.stft_matrix)
        sub_shi = np.roll(a=shi, shift=1, axis=1)
        sub_shi[:,0] = 0
        del_shi = shi - sub_shi

        N = self.stft_processor.frame_size
        Ha = self.stft_processor.hop_size

        #wk for each bin
        k = np.arange(N//2 + 1)[:, None] #2D by shape [ [k0], [k1], ... ] ready for broadcast to [ [k0, k0....], ...]
        wk = 2*np.pi*k/N

        delta = del_shi - wk*Ha
        #unwrap logic, add +pi to make it non-negative, modulo 2pi makes it bound and then -pi to get back
        delta = (delta + np.pi) % (2*np.pi) - np.pi
        wk_hat = wk + delta/Ha

        return wk_hat

    def calculate_shi_out(self, speed_factor:float = 1, pitch_factor:float = 1):

        Hs = self.get_synthesis_hop(speed_factor)
        wk_hat = self.calculate_wk_hat()
        wk_hat = wk_hat*pitch_factor

        if(self.stft_matrix is None):
            raise RuntimeError("STFT went wrong")

        shi_in = np.angle(self.stft_matrix)
        shi_out = np.zeros_like(self.stft_matrix, dtype=float) 
        shi_out[:,0] = shi_in[:, 0]
        for frame in range(1, shi_out.shape[1]):
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
    
    def vocoder_note_shift(self, semitone: float) -> np.ndarray:
        """
            semitone n shift -> freq * 2^(n/12) shift

            Pure pitch shift (duration preserved):
            1. Resample by pitch_ratio -> shifts pitch, changes length.
            2. Phase-vocoder stretch back to the original length ->
                this stage changes duration without touching pitch,
                so the pitch shift from step 1 survives intact.
        """
        if self.audio_loader.audio_data is None or self.audio_loader.sample_rate is None:
            raise RuntimeError("No audio loaded in audio_loader.")

        audio = self.audio_loader.audio_data
        sr = self.audio_loader.sample_rate
        n = len(audio)

        pitch_ratio = 2 ** (semitone / 12.0)

        # --- Step 1: resample (shifts pitch, changes length) ---
        resampled_len = max(1, round(n / pitch_ratio))
        src_positions = np.linspace(0, n - 1, resampled_len)
        resampled = np.interp(src_positions, np.arange(n), audio).astype(np.float32)

        # --- Step 2: stretch resampled audio back to original length ---
        temp_loader = AudioLoader()
        temp_loader.audio_data = resampled
        temp_loader.sample_rate = sr

        temp_vocoder = vocoder_processor(temp_loader, self.stft_processor)

        length_correction = resampled_len / n   # brings length back to n
        pitch_shifted = temp_vocoder.vocoder_process(length_correction, pitch_factor=1.0)

        return pitch_shifted

