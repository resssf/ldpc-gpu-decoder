"""AWGN Channel model"""

import numpy as np

class AWGNChannel:
    """AWGN channel"""

    def __init__(self, snr_db: float, code_rate: float = 0.5):
        """
        Args:
            snr_db: SNR in dB
            code_rate: R (for Eb/N0 conversion)
        """
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10)
        self.code_rate = code_rate

    def transmit(self, codeword: np.ndarray) -> np.ndarray:
        """Add AWGN noise

        For BPSK modulation:
        - Signal energy: E_s = 1 (per symbol)
        - Bit energy: E_b = E_s / log2(M) = 1 (BPSK, M=2)
        - SNR = E_b / (N_0 / 2) where N_0/2 is one-sided noise PSD
        - Noise std: σ = sqrt(1 / (2 * SNR_linear))
        """
        signal = Modulation.encode(codeword)  # -1, +1
        noise_std = np.sqrt(1 / (2 * self.snr_linear))
        noise = np.random.normal(0, noise_std, len(signal))
        received = signal + noise
        return received
