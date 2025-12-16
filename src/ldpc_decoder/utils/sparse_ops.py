"""Sparse matrix utilities for LDPC codes"""

import numpy as np
import torch
from scipy.sparse import csr_matrix

class SparseOps:
    """Sparse matrix operations (CSR format)"""

    @staticmethod
    def csr_to_dense(H_sparse: csr_matrix) -> np.ndarray:
        """Convert CSR to dense""" # Teta(mn + k)
        return H_sparse.toarray()

    @staticmethod
    def dense_to_csr(H_dense: np.ndarray) -> csr_matrix:
        """Convert dense to CSR"""
        return csr_matrix(H_dense) # Teta(mn)

    @staticmethod
    def csr_matvec_mod2(H: csr_matrix, v: np.ndarray) -> np.ndarray:
        """H @ v (mod 2) efficiently"""
        result = H @ v # O(nnz)
        return result % 2 # working in GF(2)

    @staticmethod
    def get_neighbors(H: csr_matrix, node_idx: int, node_type: str = 'variable'):
        """Get neighbors of variable or check node"""
        if node_type == 'variable':
            return H[:, node_idx].nonzero()[0]
        else:
            return H[node_idx, :].nonzero()[1]

    @staticmethod
    def csr_to_torch_sparse(H: csr_matrix, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Convert SciPy CSR to PyTorch sparse CSR"""
        crow_indices = torch.tensor(H.indptr, dtype=torch.int64, device=device)
        col_indices = torch.tensor(H.indices, dtype=torch.int64, device=device)
        values = torch.tensor(H.data, dtype=dtype, device=device)
        return torch.sparse_csr_tensor(crow_indices, col_indices, values, size=H.shape, device=device)

    @staticmethod
    def sparse_matvec_mod2(H_sparse_torch: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Sparse H @ v (mod 2) on GPU"""
        result = torch.sparse.mm(H_sparse_torch, v.unsqueeze(1)).squeeze() % 2
        return result
