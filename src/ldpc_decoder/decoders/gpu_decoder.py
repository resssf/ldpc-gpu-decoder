"""GPU-accelerated LDPC decoder using PyTorch"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from typing import Tuple, Optional

class MinSumDecoderGPU:
    """
    Min-Sum LDPC Decoder.
    Alghorithm normalized/offset Min-Sum алгоритм.
    """

    def __init__(
        self,
        H: csr_matrix,
        max_iterations: int = 10,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        llr_clip: float = 50.0,
        offset: float = 0.0,
        normalize_factor: float = 1.0
    ):
        """
        Args:
            H: Parity-check matrix (scipy CSR)
            max_iterations: Max iterations BP
            device: PyTorch device
            dtype: Data type
            llr_clip: LLR bound
            offset: Shift for offset Min-Sum
            normalize_factor: Normalization
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.dtype = dtype
        self.m, self.n = H.shape
        self.max_iterations = max_iterations
        self.llr_clip = llr_clip
        self.offset = offset
        self.normalize_factor = normalize_factor

        self.H = torch.tensor(H.toarray(), dtype=dtype, device=device)

        self._precompute_structure(H)

    def _precompute_structure(self, H: csr_matrix):
        """Precomputing padded arrays of neighbors and masks"""

        check_neighbors = []
        for i in range(self.m):
            start, end = H.indptr[i], H.indptr[i + 1]
            check_neighbors.append(H.indices[start:end].tolist())

        self.check_degrees = torch.tensor(
            [len(nb) for nb in check_neighbors],
            dtype=torch.long,
            device=self.device
        )
        self.max_check_deg = max(1, int(self.check_degrees.max().item()))

        # Padded neighbors ([m, max_check_deg])
        self.check_padded = torch.zeros(
            (self.m, self.max_check_deg),
            dtype=torch.long,
            device=self.device
        )
        for c, nb in enumerate(check_neighbors):
            if len(nb) > 0:
                self.check_padded[c, :len(nb)] = torch.tensor(nb, device=self.device)

        # Mask for correct positions ([m, max_check_deg])
        positions = torch.arange(self.max_check_deg, device=self.device).unsqueeze(0)
        self.check_mask = positions < self.check_degrees.unsqueeze(1)

        self.num_edges = int(self.check_mask.sum().item()) # number of correct edges

        self.mask_flat = self.check_mask.view(-1)
        self.valid_indices = self.mask_flat.nonzero(as_tuple=True)[0]

        self.edge_to_var = self.check_padded[self.check_mask]  # [num_edges]

        low_degree_checks = (self.check_degrees < 2).sum().item()
        if low_degree_checks > 0:
            print(f"Warning: {low_degree_checks} check nodes with degree < 2")

    def decode(self, llr_np: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Decoding LLR using Min-Sum.

        Args:
            llr_np: Channel LLRs, shape [n] или [batch, n]

        Returns:
            decoded: Decoded bits [batch, n]
            iterations: Number of iterations
            success: Success of decoding [batch]
        """
        llr = torch.as_tensor(llr_np, dtype=self.dtype, device=self.device)
        if llr.dim() == 1:
            llr = llr.unsqueeze(0)
        batch_size = llr.shape[0]

        llr = torch.nan_to_num(llr, nan=0.0, posinf=self.llr_clip, neginf=-self.llr_clip)
        llr = torch.clamp(llr, -self.llr_clip, self.llr_clip)

        # v2c[b, c, p] = LLR of variable check_padded[c, p]
        v2c = llr[:, self.check_padded]  # [batch, m, max_check_deg]

        converged = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        final_iter = self.max_iterations
        decoded = torch.zeros((batch_size, self.n), dtype=torch.int32, device=self.device)

        for iter_num in range(self.max_iterations):
            # Check node update
            c2v = self._check_node_update(v2c)

            # Variable node update
            v2c, marginals = self._variable_node_update(llr, c2v)

            # Hard decision
            decoded = (marginals < 0).int()

            syndrome = torch.mm(decoded.float(), self.H.t()) % 2  # [batch, m]
            new_converged = (syndrome.sum(dim=1) == 0)
            converged = converged | new_converged

            if torch.all(converged):
                final_iter = iter_num + 1
                break

        return decoded.cpu().numpy(), final_iter, converged.cpu().numpy()

    def _check_node_update(self, v2c: torch.Tensor) -> torch.Tensor:
        """
        Check node update in Min-Sum.

        For each edge (c, v):
            c2v = (signs product without v) * (min of magnidueds without v)
        """
        batch_size = v2c.shape[0]
        mask = self.check_mask.unsqueeze(0)  # [1, m, max_check_deg]

        signs = torch.sign(v2c)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        mags = torch.abs(v2c)

        mags = torch.where(mask, mags, torch.full_like(mags, float('inf')))
        signs = torch.where(mask, signs, torch.ones_like(signs))

        sign_prod = torch.prod(signs, dim=2)

        ext_signs = sign_prod.unsqueeze(2) * signs  # [batch, m, max_check_deg]

        sorted_mags, sorted_idx = torch.sort(mags, dim=2)
        min1 = sorted_mags[:, :, 0]  # [batch, m]
        min2_idx = min(1, self.max_check_deg - 1)
        min2 = sorted_mags[:, :, min2_idx]  # [batch, m]
        min1_idx = sorted_idx[:, :, 0]  # [batch, m]

        if self.offset > 0:
            min1 = torch.clamp(min1 - self.offset, min=0.0)
            min2 = torch.clamp(min2 - self.offset, min=0.0)
        if self.normalize_factor != 1.0:
            min1 = min1 / self.normalize_factor
            min2 = min2 / self.normalize_factor

        positions = torch.arange(self.max_check_deg, device=self.device)
        positions = positions.view(1, 1, -1).expand(batch_size, self.m, -1)
        min1_idx_exp = min1_idx.unsqueeze(2).expand_as(positions)

        min1_expanded = min1.unsqueeze(2).expand(-1, -1, self.max_check_deg).clone()
        min2_expanded = min2.unsqueeze(2).expand(-1, -1, self.max_check_deg)

        ext_mags = torch.where(
            positions == min1_idx_exp,
            min2_expanded,
            min1_expanded
        )

        c2v = ext_signs * ext_mags
        c2v = torch.where(mask, c2v, torch.zeros_like(c2v))
        c2v = torch.clamp(c2v, -self.llr_clip, self.llr_clip)

        return c2v

    def _variable_node_update(
        self,
        llr: torch.Tensor,
        c2v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Variable node update.

        v2c[c, v] = LLR[v] + sum_{c' != c} c2v[c', v]
                  = LLR[v] + total_c2v[v] - c2v[c, v]

        Returns:
            v2c: Updated variable-to-check message
            marginals: LLR + total_c2v for hard decision
        """
        batch_size = llr.shape[0]
        mask = self.check_mask.unsqueeze(0)

        total_c2v = torch.zeros(
            (batch_size, self.n),
            dtype=self.dtype,
            device=self.device
        )

        c2v_flat = c2v.view(batch_size, -1)  # [batch, m * max_check_deg]
        c2v_valid = c2v_flat[:, self.valid_indices]  # [batch, num_edges]

        # Scatter add for accamulation through varibles 
        var_indices_exp = self.edge_to_var.unsqueeze(0).expand(batch_size, -1)
        total_c2v.scatter_add_(1, var_indices_exp, c2v_valid)

        # Marginals for hard decision
        marginals = llr + total_c2v

        # v2c: LLR + total - self
        llr_gathered = llr[:, self.check_padded]  # [batch, m, max_check_deg]
        total_gathered = total_c2v[:, self.check_padded]

        v2c = llr_gathered + total_gathered - c2v
        v2c = torch.where(mask, v2c, torch.zeros_like(v2c))
        v2c = torch.clamp(v2c, -self.llr_clip, self.llr_clip)

        v2c = torch.nan_to_num(v2c, nan=0.0, posinf=self.llr_clip, neginf=-self.llr_clip) # checking NaN/Inf

        return v2c, marginals
