"""LDPC Code Generation and Properties"""

from typing import Tuple, Optional
from collections import deque
import numpy as np
from scipy.sparse import scr_matrix, random as sp_random

try:
    import sympy as sp_sympy
    from sympy.polys.domains import GF
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

class LDPCCodeGenerator:
    """Generate LDPC codes"""

    @staticmethod
    def generate_random_sparse(n: int, k: int, sparsity: float = 0.05, min_girth: int = 0, max_attempts: int = 10) -> csr_matrix:
        """Generate random sparse LDPC code

        Args:
            n: code length (block size)
            k: information bits
            sparsity: fraction of non-zeros in H
            min_girth: minimum required girth (if >0, regenerate until achieved)
            max_attempts: max regenerations if min_girth not met

        Returns:
            H: (n-k) × n parity check matrix
        """
        m = n - k  # number of parity check equations
        attempt = 0
        while attempt < max_attempts:
            if SPARSE_AVAILABLE:
                H = sp_random(m, n, density=sparsity, format='csr', random_state=42 + attempt)
                H.data[:] = 1  # Make binary
            else:
                # Fallback to dense
                H = np.random.rand(m, n) < sparsity
                H = H.astype(int)
            if min_girth <= 0:
                return H
            girth = LDPCCodeGenerator.calculate_girth(H)
            if girth >= min_girth:
                return H
            attempt += 1
            print(f"Attempt {attempt}: Girth {girth} < {min_girth}, regenerating...")
        return H

    @staticmethod
    def ieee_802_11_1944_rate_half() -> csr_matrix:
        """IEEE 802.11n code: n=1944, rate=0.5"""
        # Simplified: use random sparse code with similar parameters
        # In production: load from table or pyldpc library
        n, k = 1944, 972
        nnz_per_col = 3
        nnz_per_row = 6

        if SPARSE_AVAILABLE:
            H = sp_random(n - k, n, density=nnz_per_col/n, format='csr', random_state=42)
            H.data[:] = 1  # Binary matrix
        else:
            # Fallback to dense
            H_dense = np.random.rand(n - k, n) < (nnz_per_col / n)
            H_dense = H_dense.astype(int)
            H = csr_matrix(H_dense) if SPARSE_AVAILABLE else H_dense  # But since not available, keep dense
        return H

    @staticmethod
    def generate_5g_nr_bg1(Z_c: int = 384, i_LS: int = 1) -> csr_matrix:
        """Generate 5G NR LDPC base graph 1 (BG1) with lifting Z_c
        From 3GPP TS 38.212 section 5.3.2, Table 5.3.2-2
        Dimensions: m=46*Z_c, n=68*Z_c, rate varies
        Here partial implementation with example shifts; full table needs loading from file.
        """
        m_bg, n_bg = 46, 68
        H_full = np.zeros((m_bg * Z_c, n_bg * Z_c), dtype=int)

        sample_shifts = [
            (0,1,0), (0,2,135), (0,3,15), (0,4,250), (0,5,307), (0,6,73), (0,7,223),
            (1,0,96), (1,1,0), (1,2,348), (1,3,6), (1,4,138), (1,5,1), (1,6,69), (1,7,19),
            (2,0,290), (2,1,120), (2,2,0), (2,3,227), (2,4,10), (2,5,65), (2,6,210), (2,7,60),
            (3,0,159), (3,1,369), (3,2,49), (3,3,91), (3,4,186), (3,5,330), (3,6,0), (3,7,134),
        ]

        for i, j, V in sample_shifts:
            P = V % Z_c  # Actual shift
            for k in range(Z_c):
                col = (k + P) % Z_c
                H_full[i * Z_c + k, j * Z_c + col] = 1

        H = csr_matrix(H_full)
        return H

    @staticmethod
    def calculate_girth(H: csr_matrix) -> int:
        """Calculate girth (shortest cycle length) of the Tanner graph using BFS"""
        m, n = H.shape
        # Bipartite graph: variables 0 to n-1, checks n to n+m-1
        adj = [[] for _ in range(n + m)]
        for i in range(m):
            for j in (H.getrow(i).nonzero()[1] if SPARSE_AVAILABLE else np.nonzero(H[i])[0]):
                adj[j].append(n + i)
                adj[n + i].append(j)

        girth = float('inf')
        for start in range(n + m):
            dist = [-1] * (n + m)
            dist[start] = 0
            parent = [-1] * (n + m)
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth:
                            girth = cycle_len
            if girth == 4:  # Early stop for minimal girth
                return 4
        return girth if girth < float('inf') else 0  # 0 means no cycles (acyclic)

    @staticmethod
    def verify_rank(H: csr_matrix) -> Tuple[int, int, int]:
        """Verify H rank over GF(2) and return (rank, expected_rank, girth)"""
        # Girth
        girth = LDPCCodeGenerator.calculate_girth(H)

        # Rank
        H_dense = H.toarray() if SPARSE_AVAILABLE else H
        if SYMPY_AVAILABLE:
            H_matrix = sp_sympy.Matrix(H_dense)
            dom = GF(2)
            DM = DomainMatrix.from_Matrix(H_matrix, dom)
            rank = DM.rank()
        else:
            print("Warning: Using numpy.linalg for rank approximation (not exact over GF(2)).")
            rank = np.linalg.matrix_rank(H_dense % 2)  # Mod2, but SVD over floats
        expected_rank = H.shape[0]  # Should be full rank
        return rank, expected_rank, girth
    # https://github.com/Lcrypto/classic-PEG-
