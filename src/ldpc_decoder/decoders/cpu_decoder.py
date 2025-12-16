"""CPU LDPC decoder with Numba optimization"""

from numba import njit, prange
from scipy.sparse import csr_matrix
import numpy as np
from typing import Tuple

@njit(parallel=True)
def check_node_update_cpu(m: int, check_neighbors: list, v2c: np.ndarray) -> np.ndarray:
    b = v2c.shape[0]
    c2v = np.zeros_like(v2c)
    for i in prange(m):
        neighbors = check_neighbors[i]
        if len(neighbors) < 2:
            continue
        for bb in range(b):
            v_vals = v2c[bb, neighbors]
            v_abs = np.abs(v_vals)
            sorted_idx = np.argsort(v_abs)
            m1_idx = sorted_idx[0]
            m2_idx = sorted_idx[1]
            m1 = v_abs[m1_idx]
            m2 = v_abs[m2_idx]
            s = np.prod(np.sign(v_vals))
            for idx in range(len(neighbors)):
                j = neighbors[idx]
                if idx == m1_idx:
                    c2v[bb, j] = s * m2
                else:
                    c2v[bb, j] = s * m1
    return c2v

@njit(parallel=True)
def variable_node_update_cpu(n: int, var_neighbors: list, llr: np.ndarray, c2v: np.ndarray) -> np.ndarray:
    b = llr.shape[0]
    v2c_new = np.zeros_like(llr)
    for j in prange(n):
        neighbors = var_neighbors[j]
        for bb in range(b):
            v2c_new[bb, j] = llr[bb, j] + np.sum(c2v[bb, neighbors])
    v2c_new = np.clip(v2c_new, -100, 100)
    return v2c_new

class MinSumDecoderCPU:
    def __init__(self, H: csr_matrix, max_iterations: int = 10):
        self.m, self.n = H.shape
        self.max_iterations = max_iterations
        self.H = H
        # Precompute neighbors as list of np.array for Numba
        self.check_neighbors = [np.array(self.H.getrow(i).nonzero()[1]) for i in range(self.m)]
        self.var_neighbors = [np.array(self.H.getcol(j).nonzero()[0]) for j in range(self.n)]

    def decode(self, llr_np: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        if llr_np.ndim == 1:
            llr_np = llr_np[None, :]  # Add batch dim
        batch_size = llr_np.shape[0]
        v2c = llr_np.copy().astype(np.float32)
        converged = np.zeros(batch_size, dtype=bool)
        iterations = self.max_iterations
        for it in range(self.max_iterations):
            c2v = check_node_update_cpu(self.m, self.check_neighbors, v2c)
            v2c = variable_node_update_cpu(self.n, self.var_neighbors, llr_np, c2v)
            decoded = (v2c < 0).astype(int)
            syndrome = np.zeros((batch_size, self.m), dtype=int)
            for b in range(batch_size):
                syndrome[b] = (self.H @ decoded[b]) % 2
            new_converged = np.all(syndrome == 0, axis=1)
            converged |= new_converged
            if np.all(converged):
                iterations = it + 1
                break
        success = converged
        return decoded, iterations, success
