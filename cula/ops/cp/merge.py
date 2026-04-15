# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Merge step for Intra-Card Context Parallel chunk_delta_h.

Implements the prefix-scan merge:
    For each original sequence split into sub-sequences [s0, s1, ..., s_{n-1}]:
        h0_s0 = initial_state (or zero)
        h0_s1 = m_s0 @ h0_s0 + he_s0
        h0_s2 = m_s1 @ h0_s1 + he_s1
        ...
    Produces initial states for all non-first sub-sequences.

Uses a single Triton kernel launch instead of a Python loop of baddbmm calls.
The kernel parallelizes across (V-blocks, split-sequences, heads) while the
prefix scan within each split-sequence is sequential inside the kernel.

Input:  hm [S_split, H, K, V+K] fp32 — packed (he, m) from pre_scan
Output: h  [num_non_first, H, K, V] fp32 — initial states for non-first sub-seqs
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BV': 32}, num_warps=4),
        triton.Config({'BV': 64}, num_warps=4),
        triton.Config({'BV': 32}, num_warps=2),
        triton.Config({'BV': 64}, num_warps=2),
    ],
    key=['K', 'V'],
)
@triton.jit
def merge_fwd_kernel(
    hm_ptr,                 # [S, H, K, V+K] fp32
    h_out_ptr,              # [num_non_first, H, K, V] fp32
    h0_ptr,                 # [N, H, K, V] fp32 or None
    seq_starts_ptr,         # [num_split_seqs] int32 — start subseq index per split seq
    seq_counts_ptr,         # [num_split_seqs] int32 — num subseqs per split seq
    init_offsets_ptr,       # [num_split_seqs + 1] int32
    split_seq_ids_ptr,      # [num_split_seqs] int32
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    HAS_H0: tl.constexpr,
    NUM_SPLIT_SEQS,
):
    """Single-launch Triton merge kernel.

    Grid: (V // BV, NUM_SPLIT_SEQS, H)
    Each program handles one (v_block, split_seq, head) and loops
    over sub-sequences sequentially for the prefix scan.
    """
    i_v = tl.program_id(0)
    i_seq = tl.program_id(1)
    i_h = tl.program_id(2)

    if i_seq >= NUM_SPLIT_SEQS:
        return

    ss_start = tl.load(seq_starts_ptr + i_seq).to(tl.int32)
    n_ss = tl.load(seq_counts_ptr + i_seq).to(tl.int32)
    init_base = tl.load(init_offsets_ptr + i_seq).to(tl.int32)

    VK: tl.constexpr = V + K
    stride_hm_s = H * K * VK
    stride_hm_h = K * VK

    # Initialize h from h0 or zeros
    if HAS_H0:
        orig_id = tl.load(split_seq_ids_ptr + i_seq).to(tl.int32)
        h0_base = orig_id * H * K * V + i_h * K * V
        p_h0 = tl.make_block_ptr(
            h0_ptr + h0_base, (K, V), (V, 1),
            (0, i_v * BV), (K, BV), (1, 0)
        )
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_h = tl.zeros([K, BV], dtype=tl.float32)

    for idx in range(n_ss):
        i_ss = ss_start + idx
        base = i_ss * stride_hm_s + i_h * stride_hm_h

        # Load he: hm[i_ss, i_h, :, i_v*BV : (i_v+1)*BV]
        p_he = tl.make_block_ptr(
            hm_ptr + base, (K, V), (VK, 1),
            (0, i_v * BV), (K, BV), (1, 0)
        )
        b_he = tl.load(p_he, boundary_check=(0, 1)).to(tl.float32)

        # Load m: hm[i_ss, i_h, :, V:V+K]
        p_m = tl.make_block_ptr(
            hm_ptr + base + V, (K, K), (VK, 1),
            (0, 0), (K, K), (1, 0)
        )
        b_m = tl.load(p_m, boundary_check=(0, 1)).to(tl.float32)

        # h = m @ h + he  (tf32 Tensor Core: ~7e-4 rel per step, ~1.7x slower than bf16
        # but much better precision for scan chain accumulation)
        b_h = tl.dot(b_m, b_h, input_precision='tf32') + b_he

        # Store for non-first sub-seqs (idx 0..n_ss-2)
        if idx < n_ss - 1:
            out_idx = init_base + idx
            out_base = out_idx * H * K * V + i_h * K * V
            p_out = tl.make_block_ptr(
                h_out_ptr + out_base, (K, V), (V, 1),
                (0, i_v * BV), (K, BV), (1, 0)
            )
            tl.store(p_out, b_h.to(p_out.dtype.element_ty), boundary_check=(0, 1))


# Cache for GPU metadata tensors to avoid repeated H2D copies
_merge_meta_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def merge_fwd(
    hm: torch.Tensor,
    seq_starts: list[int],
    seq_counts: list[int],
    init_offsets: list[int],
    split_seq_ids: list[int],
    h0: torch.Tensor | None,
    num_non_first: int,
) -> torch.Tensor:
    """Prefix-scan merge using a single Triton kernel launch.

    Args:
        hm:             [S_split, H, K, V+K] fp32 — packed (he, m) from pre_scan
        seq_starts:     Start subseq index per split sequence (len = num_split_seqs)
        seq_counts:     Num subseqs per split sequence (len = num_split_seqs)
        init_offsets:   Cumulative non-first counts per split sequence (len = num_split_seqs+1)
        split_seq_ids:  Original seq index for each split sequence
        h0:             [N, H, K, V] fp32 or None
        num_non_first:  Total non-first sub-sequences

    Returns:
        initial_states_merge: [num_non_first, H, K, V] fp32
    """
    _, H, K, VK = hm.shape
    V = VK - K
    device = hm.device
    num_split_seqs = len(split_seq_ids)

    h_out = hm.new_empty(num_non_first, H, K, V)

    # Convert metadata lists to GPU tensors (cached)
    cache_key = (tuple(seq_starts), tuple(seq_counts), tuple(init_offsets), tuple(split_seq_ids))
    cached = _merge_meta_cache.get(cache_key)
    if cached is not None:
        starts_gpu, counts_gpu, init_off_gpu, sid_gpu = cached
    else:
        starts_gpu = torch.tensor(seq_starts, dtype=torch.int32, device=device)
        counts_gpu = torch.tensor(seq_counts, dtype=torch.int32, device=device)
        init_off_gpu = torch.tensor(init_offsets, dtype=torch.int32, device=device)
        sid_gpu = torch.tensor(split_seq_ids, dtype=torch.int32, device=device)
        _merge_meta_cache[cache_key] = (starts_gpu, counts_gpu, init_off_gpu, sid_gpu)
        # Evict old entries
        if len(_merge_meta_cache) > 32:
            oldest = next(iter(_merge_meta_cache))
            del _merge_meta_cache[oldest]

    grid = lambda meta: (V // meta['BV'], num_split_seqs, H)
    merge_fwd_kernel[grid](
        hm, h_out,
        h0 if h0 is not None else hm,  # dummy ptr when no h0
        starts_gpu, counts_gpu, init_off_gpu, sid_gpu,
        H=H, K=K, V=V,
        HAS_H0=h0 is not None,
        NUM_SPLIT_SEQS=num_split_seqs,
    )

    return h_out
