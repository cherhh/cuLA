# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Intra-Card Context Parallel (CP) for Chunk Delta H — cuLA Blackwell Implementation.

Overview:
    Long sequences on a single card are split into sub-sequences, each processed
    independently via cuLA's CuTeDSL chunk_delta_h kernel. A prefix-scan merge
    step propagates initial states across sub-sequences, eliminating the sequential
    bottleneck of the original single-pass recurrence.

Pipeline (3 stages):
    1. Pre-Scan: For each sub-sequence, compute packed (he, m) state:
         he [K, V] = cumulative delta-rule update (the "h-exit" state)
         m  [K, K] = cumulative decay matrix
       Packed as hm [S_split, H, K, K+V] where columns [0:V]=he, [V:V+K]=m

    2. Merge: Prefix scan across sub-sequences of the same original sequence.
       For sub-sequence j:  h0_j = m_j @ h0_{j-1} + he_j
       Produces per-sub-sequence initial states.

    3. Forward H: Run cuLA's existing chunk_gated_delta_rule_fwd_h on the
       split sub-sequences with the merged initial states.

Reference:
    - FLA intra-card CP: fla/ops/common/intracard_cp.py
    - FLA CP kernels:    fla/ops/cp/chunk_delta_h.py
    - cuLA chunk_delta_h: cula/ops/chunk_delta_h.py
"""

from __future__ import annotations

import weakref
from collections import OrderedDict
from typing import NamedTuple

import torch

# Lazy import to avoid circular dependency:
# cula.ops.cp.chunk_delta_h ↔ cula.ops.chunk_delta_h
_chunk_gated_delta_rule_fwd_h = None


def _get_fwd_h():
    global _chunk_gated_delta_rule_fwd_h
    if _chunk_gated_delta_rule_fwd_h is None:
        from cula.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h
        _chunk_gated_delta_rule_fwd_h = chunk_gated_delta_rule_fwd_h
    return _chunk_gated_delta_rule_fwd_h


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class SplitSeqInfo(NamedTuple):
    """Metadata for sequences that were split into sub-sequences.

    Attributes:
        split_seq_ids:    List of original sequence indices that were split.
        start_subseq_idx: For each split seq, the index of its first sub-seq
                          in the expanded cu_seqlens array.
        num_subseqs:      Number of sub-sequences for each split sequence.
    """
    split_seq_ids: list[int]
    start_subseq_idx: list[int]
    num_subseqs: list[int]

    @property
    def num_split_seqs(self) -> int:
        return len(self.split_seq_ids)


class _CacheEntry(NamedTuple):
    """Cached precomputed indices and GPU tensors for a given cu_seqlens input.

    Caching avoids redundant CPU work + H2D transfers across calls with the
    same sequence layout (common in vLLM continuous batching).
    """
    cu_seqlens_ref: weakref.ref          # weakref to original cu_seqlens tensor
    cu_seqlens_subseq_values: list[int]  # CPU list of split boundaries
    split_info: SplitSeqInfo
    total_subseqs: int
    non_first_indices: torch.Tensor       # [num_non_first] int64 GPU
    first_subseq_indices: torch.Tensor    # [N_orig] int64 GPU
    last_subseq_indices: torch.Tensor     # [N_orig] int64 GPU
    num_non_first: int
    merge_seq_starts: list[int]
    merge_seq_counts: list[int]
    merge_init_offsets: list[int]
    # GPU tensors (kept alive to avoid repeated H2D)
    cu_seqlens_subseq_gpu: torch.Tensor


# LRU cache for precomputed indices
_intracard_cache: OrderedDict[tuple, _CacheEntry] = OrderedDict()
_INTRACARD_CACHE_MAXSIZE = 8

# Cached device SM count to avoid repeated get_device_properties calls
_device_num_sms: dict[torch.device, int] = {}


def _get_num_sms(device: torch.device) -> int:
    """Get SM count, cached per device."""
    if device not in _device_num_sms:
        _device_num_sms[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _device_num_sms[device]


# ---------------------------------------------------------------------------
# Stage 0: Sequence splitting
# ---------------------------------------------------------------------------

def compute_subseq_len(
    seq_len: int,
    num_sms: int,
    num_heads: int,
    chunk_size: int = 64,
) -> int:
    """Determine optimal sub-sequence length for splitting.

    Heuristic: choose split count to saturate SMs with a single long sequence.
    cuLA's chunk_delta_h has grid = (num_v_blocks, N*H) where num_v_blocks=2
    (V=128, BV=64). Each sub-seq contributes 2*H blocks.

    A minimum sub-sequence length prevents over-splitting short sequences
    in mixed-length batches.

    Args:
        seq_len:   Length of the longest sequence in the batch.
        num_sms:   Number of SMs on the device.
        num_heads: Number of attention heads.
        chunk_size: Chunk size (default 64).

    Returns:
        Sub-sequence length (multiple of chunk_size).
    """
    seq_chunks = (seq_len + chunk_size - 1) // chunk_size

    if seq_chunks < 8:
        return seq_len

    # Target splits: saturate SMs with the longest sequence alone.
    # cuLA kernel grid = (NUM_V_BLOCKS, N*H), occ=1.
    NUM_V_BLOCKS = 2
    target_splits = max(4, num_sms // (NUM_V_BLOCKS * num_heads))

    subseq_chunks = (seq_chunks + target_splits - 1) // target_splits

    # Floor: prevent over-splitting short sequences in mixed-length batches.
    # MIN_SUBSEQ_CHUNKS=128 → subseq_len >= 8192 tokens,
    # split threshold (3 * subseq_len) = 24576 tokens.
    MIN_SUBSEQ_CHUNKS = 128
    subseq_chunks = max(subseq_chunks, MIN_SUBSEQ_CHUNKS)

    return subseq_chunks * chunk_size


def prepare_subseq_cu_seqlens(
    cu_seqlens_cpu: torch.Tensor,
    subseq_len: int,
    chunk_size: int = 64,
    max_splits: int = 32,
) -> tuple[list[int], SplitSeqInfo | bool, int]:
    """Insert sub-sequence split points into the original cu_seqlens.

    For each sequence longer than a threshold, split it into evenly sized
    sub-sequences (each a multiple of chunk_size). Short sequences are
    kept intact.

    Args:
        cu_seqlens_cpu: Original cu_seqlens on CPU, shape [N+1].
        subseq_len:     Target sub-sequence length (from compute_subseq_len).
        chunk_size:     Chunk size.
        max_splits:     Maximum number of splits per sequence.

    Returns:
        boundaries:    List[int] — expanded cu_seqlens with split points.
        split_info:    SplitSeqInfo or False if no splitting was needed.
        total_subseqs: Total number of sub-sequences after splitting.
    """
    N = len(cu_seqlens_cpu) - 1
    if N == 0:
        return cu_seqlens_cpu.tolist(), False, 0

    subseq_chunks = (subseq_len + chunk_size - 1) // chunk_size
    threshold_subseq_len = 3 * subseq_len

    split_seq_ids: list[int] = []
    start_subseq_idxs: list[int] = []
    num_subseqs_list: list[int] = []

    boundaries: list[int] = [0]
    cumsum_offset = 0

    for i in range(N):
        seq_start = int(cu_seqlens_cpu[i].item())
        seq_end = int(cu_seqlens_cpu[i + 1].item())
        seq_len_i = seq_end - seq_start
        seq_chunks_i = (seq_len_i + chunk_size - 1) // chunk_size

        if seq_len_i >= threshold_subseq_len:
            num_ss = min(max_splits, (seq_chunks_i + subseq_chunks - 1) // subseq_chunks)
            chunks_per = (seq_chunks_i + num_ss - 1) // num_ss
            actual_ssl = chunks_per * chunk_size

            split_seq_ids.append(i)
            start_subseq_idxs.append(cumsum_offset)
            num_subseqs_list.append(num_ss)

            for j in range(num_ss):
                boundary = min(seq_start + (j + 1) * actual_ssl, seq_end)
                boundaries.append(boundary)
            cumsum_offset += num_ss
        else:
            boundaries.append(seq_end)
            cumsum_offset += 1

    if not split_seq_ids:
        return cu_seqlens_cpu.tolist(), False, 0

    total_subseqs = cumsum_offset
    split_info = SplitSeqInfo(
        split_seq_ids=split_seq_ids,
        start_subseq_idx=start_subseq_idxs,
        num_subseqs=num_subseqs_list,
    )
    return boundaries, split_info, total_subseqs


# ---------------------------------------------------------------------------
# Stage 0b: Index precomputation
# ---------------------------------------------------------------------------

def _precompute_intracard_indices(
    split_info: SplitSeqInfo,
    cu_seqlens_subseq_values: list[int],
    N_orig: int,
) -> tuple[list[int], int, list[int], list[int], list[int], int, list[int], list[int]]:
    """Precompute all derived scatter/gather indices from split metadata.

    Returns (all pure Python lists):
        non_first_indices:      Indices for scattering merged initial states.
        first_subseq_indices:   Indices of first sub-seq per original sequence.
        last_subseq_indices:    Indices of last sub-seq per original sequence.
        num_non_first:          Count of non-first sub-sequences (merge work items).
        merge_seq_starts:       Start subseq index per split sequence.
        merge_seq_counts:       Num subseqs per split sequence.
        merge_init_offsets:     Cumulative non-first counts for merge kernel.
    """
    starts = split_info.start_subseq_idx
    num_ss = split_info.num_subseqs
    split_ids = split_info.split_seq_ids

    # Per-original-sequence sub-seq count (default 1 for unsplit)
    num_subseqs_per_seq = [1] * N_orig
    for sid, nss in zip(split_ids, num_ss):
        num_subseqs_per_seq[sid] = nss

    # Non-first indices: where to scatter merged initial states
    non_first_indices: list[int] = []
    for s, n in zip(starts, num_ss):
        for j in range(1, n):
            non_first_indices.append(s + j)

    # First sub-seq indices: where to scatter original h0
    first_subseq_indices: list[int] = [0]
    running = 0
    for i in range(N_orig - 1):
        running += num_subseqs_per_seq[i]
        first_subseq_indices.append(running)

    # Last sub-seq indices: where to gather final states
    last_subseq_indices: list[int] = []
    running = 0
    for n in num_subseqs_per_seq:
        running += n
        last_subseq_indices.append(running - 1)

    # Merge kernel metadata — per-split-sequence start index and count
    # (NOT CSR offsets, because split sequences' subseqs may be non-contiguous
    # in the hm tensor when there are unsplit sequences in between)
    merge_seq_starts: list[int] = list(starts)
    merge_seq_counts: list[int] = list(num_ss)
    merge_init_offsets: list[int] = [0]
    for n in num_ss:
        merge_init_offsets.append(merge_init_offsets[-1] + n - 1)
    num_non_first = merge_init_offsets[-1]

    return (
        non_first_indices,
        first_subseq_indices,
        last_subseq_indices,
        num_non_first,
        merge_seq_starts,
        merge_seq_counts,
        merge_init_offsets,
    )


# ---------------------------------------------------------------------------
# Stage 1: Pre-Scan kernel
# ---------------------------------------------------------------------------

def intracard_pre_scan(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor | None,
    cu_seqlens_subseq_split: torch.Tensor,
    S_split: int,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compute packed (he, m) state for each split sub-sequence.

    For each sub-sequence s in [0, S_split):
        Run the delta-rule recurrence across all chunks, accumulating:
          he[s, h, :K, :V]  — exit h-state (K×V)
          m[s, h, :K, :K]   — cumulative decay matrix (K×K)

    Packed output: hm [S_split, H, K, V+K]
        columns [0:V]   = he (K×V state)
        columns [V:V+K] = m  (K×K decay matrix)

    Uses cuLA's CuTeDSL Blackwell kernel for he and a Triton kernel for m,
    both launched concurrently on separate CUDA streams.

    Args:
        k:  [1, T, H, K] bf16 (varlen packed)
        w:  [1, T, H, K] bf16
        u:  [1, T, H, V] bf16
        gk: [1, T, H, K] fp32 or None
        cu_seqlens_subseq_split: [S_split+1] int32, boundaries for split sub-seqs only
        S_split: number of split sub-sequences
        chunk_size: chunk size (default 64)

    Returns:
        hm: [S_split, H, K, V+K] fp32
    """
    from cula.ops.cp.pre_scan import chunk_delta_rule_pre_scan

    return chunk_delta_rule_pre_scan(
        k=k, 
        w=w, 
        u=u, 
        gk=gk,
        cu_seqlens_split=cu_seqlens_subseq_split,
        S_split=S_split,
        chunk_size=chunk_size,
    )


# ---------------------------------------------------------------------------
# Stage 2: Merge (prefix scan across sub-sequences)
# ---------------------------------------------------------------------------

def intracard_merge(
    hm: torch.Tensor,
    split_info: SplitSeqInfo,
    num_non_first: int,
    merge_seq_starts: list[int],
    merge_seq_counts: list[int],
    merge_init_offsets: list[int],
    device: torch.device,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, int]:
    """Merge sub-sequence states via prefix scan to produce initial states.

    For each original sequence that was split into [s0, s1, ..., s_{n-1}]:
        h0_s0 = initial_state (or zero)
        h0_s1 = m_s0 @ h0_s0 + he_s0
        h0_s2 = m_s1 @ h0_s1 + he_s1
        ...

    Args:
        hm:                  [S_split, H, K, V+K] fp32 — packed (he, m) from pre_scan
        split_info:          SplitSeqInfo metadata
        num_non_first:       Number of non-first sub-sequences to produce states for
        merge_seq_starts:    Start subseq index per split sequence
        merge_seq_counts:    Num subseqs per split sequence
        merge_init_offsets:  Cumulative non-first counts per split sequence
        device:              CUDA device
        initial_state:       [N, H, K, V] fp32 or None — original initial states

    Returns:
        initial_states_merge: [num_non_first, H, K, V] fp32, or None if no merge needed
        num_non_first:        Number of produced initial states
    """
    from cula.ops.cp.merge import merge_fwd

    if num_non_first == 0:
        return None, 0

    H = hm.shape[1]
    K = hm.shape[2]
    V = hm.shape[3] - K

    initial_states_merge = merge_fwd(
        hm=hm,
        seq_starts=merge_seq_starts,
        seq_counts=merge_seq_counts,
        init_offsets=merge_init_offsets,
        split_seq_ids=split_info.split_seq_ids,
        h0=initial_state,
        num_non_first=num_non_first,
    )

    return initial_states_merge, num_non_first


# ---------------------------------------------------------------------------
# Stage 3: Scatter initial states + run forward h
# ---------------------------------------------------------------------------

def _scatter_initial_states(
    initial_state: torch.Tensor | None,
    initial_states_merge: torch.Tensor | None,
    num_non_first: int,
    total_subseqs: int,
    first_subseq_indices: torch.Tensor,
    non_first_indices: torch.Tensor,
    H: int,
    K: int,
    V: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the expanded initial_state tensor for all sub-sequences.

    initial_state_expanded[first_subseq_indices] = initial_state (original h0)
    initial_state_expanded[non_first_indices]    = initial_states_merge (from merge)

    Args & Returns: self-explanatory from types.

    Returns:
        initial_state_expanded: [total_subseqs, H, K, V] fp32
    """
    initial_state_expanded = torch.zeros(
        total_subseqs, H, K, V, device=device, dtype=torch.float32
    )

    if initial_state is not None:
        initial_state_expanded[first_subseq_indices] = initial_state

    if initial_states_merge is not None and num_non_first > 0:
        initial_state_expanded[non_first_indices] = initial_states_merge

    return initial_state_expanded


def _gather_final_states(
    final_state_subseq: torch.Tensor | None,
    last_subseq_indices: torch.Tensor,
    output_final_state: bool,
) -> torch.Tensor | None:
    """Gather final h-state from last sub-sequence of each original sequence.

    Args:
        final_state_subseq: [total_subseqs, H, K, V] fp32, from fwd_h
        last_subseq_indices: Indices of last sub-seq per original sequence
        output_final_state: Whether final state was requested

    Returns:
        final_state: [N, H, K, V] fp32, or None
    """
    if not output_final_state or final_state_subseq is None:
        return None
    return final_state_subseq[last_subseq_indices]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def intracard_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    max_splits: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Intra-card context-parallel chunk_delta_h forward using cuLA's Blackwell kernel.

    Splits long sequences into sub-sequences, runs pre_scan + merge to compute
    per-sub-sequence initial states, then dispatches cuLA's chunk_gated_delta_rule_fwd_h.

    API is a drop-in replacement for chunk_gated_delta_rule_fwd_h with varlen.

    Pipeline:
        1. Determine subseq_len from seq lengths and SM count
        2. Split cu_seqlens → expanded cu_seqlens_subseq
        3. If no split needed → early return to non-CP path
        4. Pre-scan: compute (he, m) for split sub-sequences
        5. Merge: prefix scan → initial states for non-first sub-sequences
        6. Scatter: build initial_state_expanded
        7. Run chunk_gated_delta_rule_fwd_h on split sub-sequences
        8. Gather: extract final states from last sub-sequences

    Args:
        k:  [1, T, H, K]  bf16 (varlen packed)
        w:  [1, T, H, K]  bf16
        u:  [1, T, H, V]  bf16
        g:  [1, T, H]     fp32 or None (scalar gate, currently unused in cuLA)
        gk: [1, T, H, K]  fp32 or None (key gate)
        initial_state:     [N, H, K, V] fp32 or None
        output_final_state: bool
        chunk_size:        64
        save_new_value:    whether to return v_new
        cu_seqlens:        [N+1] int64/int32 — REQUIRED for intra-card CP
        cu_seqlens_cpu:    [N+1] CPU tensor (optional, avoids D2H copy)
        chunk_indices:     [NT, 2] int32 (optional, auto-computed if None)
        max_splits:        Maximum splits per sequence

    Returns:
        h:           [1, NT, H, K, V] bf16
        v_new:       [1, T, H, V] bf16 or None
        final_state: [N, H, K, V] fp32 or None
    """
    assert cu_seqlens is not None, "intracard_fwd_h requires cu_seqlens (varlen mode)"
    assert g is None, "intracard CP does not yet support scalar gate (g); only per-key gate (gk) is supported"

    _, _, H, K = k.shape
    V = u.shape[3]
    device = k.device

    if cu_seqlens_cpu is None:
        cu_seqlens_cpu = cu_seqlens.cpu()

    seq_lens = torch.diff(cu_seqlens_cpu)
    max_seq_len = int(seq_lens.max().item())
    num_sms = _get_num_sms(device)
    subseq_len = compute_subseq_len(max_seq_len, num_sms, H, chunk_size)

    # Same as FLA: skip CP when all sequences are shorter than 2 * subseq_len.
    # With MIN_SUBSEQ_CHUNKS=128 and chunk_size=64, subseq_len >= 8192,
    # so early_return threshold >= 16384. Per-sequence split threshold in
    # prepare_subseq_cu_seqlens is 3 * subseq_len (>= 24576).
    early_return = (seq_lens < 2 * subseq_len).all().item()


    cached = None
    cache_key = None

    if not early_return:
        cache_key = (
            id(cu_seqlens), 
            subseq_len, 
            chunk_size, 
            max_splits, 
            str(device)
        )
        cached = _intracard_cache.get(cache_key)
        if cached is not None:
            if cached.cu_seqlens_ref() is cu_seqlens:
                _intracard_cache.move_to_end(cache_key)
            else:
                _intracard_cache.pop(cache_key, None)
                cached = None

        if cached is None:
            cu_seqlens_subseq_values, split_info, total_subseqs = prepare_subseq_cu_seqlens(
                cu_seqlens_cpu, subseq_len, chunk_size, max_splits=max_splits
            )
        else:
            split_info = cached.split_info

    
    if early_return or not split_info:
        return _get_fwd_h()(
            k=k, w=w, u=u, g=g, gk=gk,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            save_new_value=save_new_value,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            _no_cp=True,
        )

    N_orig = len(cu_seqlens_cpu) - 1

    
    if cached is not None:
        cu_seqlens_subseq_values = cached.cu_seqlens_subseq_values
        total_subseqs = cached.total_subseqs
        non_first_indices = cached.non_first_indices
        first_subseq_indices = cached.first_subseq_indices
        last_subseq_indices = cached.last_subseq_indices
        num_non_first = cached.num_non_first
        merge_seq_starts = cached.merge_seq_starts
        merge_seq_counts = cached.merge_seq_counts
        merge_init_offsets = cached.merge_init_offsets
        cu_seqlens_subseq_gpu = cached.cu_seqlens_subseq_gpu
    else:
        (
            non_first_indices,
            first_subseq_indices,
            last_subseq_indices,
            num_non_first,
            merge_seq_starts,
            merge_seq_counts,
            merge_init_offsets,
        ) = _precompute_intracard_indices(split_info, cu_seqlens_subseq_values, N_orig)

        # Convert scatter/gather index lists to GPU tensors to avoid
        # per-call CPU→GPU copies from Python list advanced indexing.
        non_first_indices = torch.tensor(non_first_indices, dtype=torch.int64, device=device)
        first_subseq_indices = torch.tensor(first_subseq_indices, dtype=torch.int64, device=device)
        last_subseq_indices = torch.tensor(last_subseq_indices, dtype=torch.int64, device=device)

        cu_seqlens_subseq_gpu = torch.tensor(
            cu_seqlens_subseq_values, dtype=cu_seqlens_cpu.dtype, device=device
        )

        # Store in cache
        _intracard_cache[cache_key] = _CacheEntry(
            cu_seqlens_ref=weakref.ref(cu_seqlens),
            cu_seqlens_subseq_values=cu_seqlens_subseq_values,
            split_info=split_info,
            total_subseqs=total_subseqs,
            non_first_indices=non_first_indices,
            first_subseq_indices=first_subseq_indices,
            last_subseq_indices=last_subseq_indices,
            num_non_first=num_non_first,
            merge_seq_starts=merge_seq_starts,
            merge_seq_counts=merge_seq_counts,
            merge_init_offsets=merge_init_offsets,
            cu_seqlens_subseq_gpu=cu_seqlens_subseq_gpu,
        )
        while len(_intracard_cache) > _INTRACARD_CACHE_MAXSIZE:
            _intracard_cache.popitem(last=False)

    
    # Compute packed (he, m) for ALL sub-sequences (including unsplit ones).
    # We run pre_scan on the full expanded cu_seqlens to avoid non-contiguous
    # sub-sequence boundary issues.
    hm = intracard_pre_scan(
        k=k, 
        w=w, 
        u=u, 
        gk=gk,
        cu_seqlens_subseq_split=cu_seqlens_subseq_gpu,
        S_split=total_subseqs,
        chunk_size=chunk_size,
    )

    
    # Prefix scan across sub-sequences → initial states for non-first sub-seqs
    initial_states_merge, num_non_first = intracard_merge(
        hm=hm,
        split_info=split_info,
        num_non_first=num_non_first,
        merge_seq_starts=merge_seq_starts,
        merge_seq_counts=merge_seq_counts,
        merge_init_offsets=merge_init_offsets,
        device=device,
        initial_state=initial_state,
    )

    
    initial_state_expanded = _scatter_initial_states(
        initial_state=initial_state,
        initial_states_merge=initial_states_merge,
        num_non_first=num_non_first,
        total_subseqs=total_subseqs,
        first_subseq_indices=first_subseq_indices,
        non_first_indices=non_first_indices,
        H=H, K=K, V=V, device=device,
    )

    
    h, v_new, final_state_subseq = _get_fwd_h()(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=initial_state_expanded,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=save_new_value,
        cu_seqlens=cu_seqlens_subseq_gpu,
        _no_cp=True,
    )

    
    final_state = _gather_final_states(
        final_state_subseq=final_state_subseq,
        last_subseq_indices=last_subseq_indices,
        output_final_state=output_final_state,
    )

    return h, v_new, final_state
