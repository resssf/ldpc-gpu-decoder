class Modulation:
    """BPSK modulation"""

    @staticmethod
    def encode(bits: np.ndarray) -> np.ndarray:
        """Binary phase-shift keying: 0,1 → -1,+1"""
        return 1 - 2 * bits

    @staticmethod
    def decode(symbols: np.ndarray) -> np.ndarray:
        """Hard decision"""
        return (symbols > 0).astype(int)
