"""Validation methods for LDPC decoders"""

import numpy as np
from scipy.sparse import csr_matrix

class Validator:
    """Correctness validation (multiple methods)"""

    @staticmethod
    def method1_syndrome_check(H: csr_matrix, codeword: np.ndarray) -> bool:
        """Method 1: H * c = 0 (mod 2)"""
        syndrome = SparseOps.csr_matvec_mod2(H, codeword)
        return np.all(syndrome == 0)

    @staticmethod
    def method2_known_codeword(decoded: np.ndarray, transmitted: np.ndarray) -> bool:
        """Method 2: Exact recovery at high SNR"""
        return np.array_equal(decoded, transmitted)

    @staticmethod
    def method3_cpu_gpu_agreement(cpu_result: np.ndarray, gpu_result: np.ndarray, tolerance: float = 0) -> bool:
        """Method 3: CPU vs GPU should match (exactly for discrete)"""
        return np.array_equal(cpu_result, gpu_result)

    @staticmethod
    def method4_ber_reasonable(true_bits: np.ndarray, decoded_bits: np.ndarray, max_ber: float = 0.2) -> bool:
        """Method 4: BER should be reasonable"""
        errors = np.sum(true_bits != decoded_bits)
        ber = errors / len(true_bits)
        return ber < max_ber
