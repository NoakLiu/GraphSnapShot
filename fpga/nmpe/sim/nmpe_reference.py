"""
Behavioral model of the FPGA NMPE vector path (paper §4.3):
SEMHS slab edges -> neighbor IDs -> feature gather -> per-seed mean aggregation.

Run: python nmpe_reference.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

NodeId = int


@dataclass
class SlabHop:
    """One hop slab: directed edges (src -> dst) stored contiguously for DMA."""

    hop: int
    edges: List[Tuple[NodeId, NodeId]]


def group_neighbors_by_seed(
    edges: Sequence[Tuple[NodeId, NodeId]],
) -> Dict[NodeId, List[NodeId]]:
    nbrs: Dict[NodeId, List[NodeId]] = {}
    for s, d in edges:
        nbrs.setdefault(s, []).append(d)
    return nbrs


def mean_aggregate(
    X: np.ndarray,
    seed: NodeId,
    neighbors: Sequence[NodeId],
) -> np.ndarray:
    """IAU: mean of feature rows for neighbors of one seed (empty -> zeros)."""
    if not neighbors:
        return np.zeros(X.shape[1], dtype=X.dtype)
    idx = np.asarray(neighbors, dtype=np.int64)
    return X[idx].mean(axis=0)


def vre_gather(X: np.ndarray, node_ids: Sequence[NodeId]) -> np.ndarray:
    """VRE: batch feature lookup X[node_ids]."""
    idx = np.asarray(list(node_ids), dtype=np.int64)
    return X[idx]


def run_nmpe_hop(
    X: np.ndarray,
    slab_edges: Sequence[Tuple[NodeId, NodeId]],
    seeds: Sequence[NodeId] | None = None,
) -> Dict[NodeId, np.ndarray]:
    """
    Full NMPE behavior for one hop slab: for each seed appearing in the slab,
    return mean neighbor features (matches mean-aggregation message reduction).
    """
    grouped = group_neighbors_by_seed(slab_edges)
    if seeds is not None:
        want = set(seeds)
        grouped = {s: grouped[s] for s in want if s in grouped}
    return {s: mean_aggregate(X, s, grouped[s]) for s in grouped}


def dma_slabs_sequential(slabs: Mapping[int, SlabHop]) -> List[Tuple[int, Tuple[NodeId, NodeId]]]:
    """Order slabs by hop (1..k) as the DMA controller would stream."""
    out: List[Tuple[int, Tuple[NodeId, NodeId]]] = []
    for h in sorted(slabs.keys()):
        for e in slabs[h].edges:
            out.append((h, e))
    return out


def _self_test() -> None:
    rng = np.random.default_rng(0)
    n, d = 100, 8
    X = rng.standard_normal((n, d)).astype(np.float32)

    # Toy 1-hop slab: seed 1 -> {2,3}, seed 4 -> {5}
    edges = [(1, 2), (1, 3), (4, 5)]
    msgs = run_nmpe_hop(X, edges)
    np.testing.assert_allclose(msgs[1], (X[2] + X[3]) / 2.0, rtol=1e-5)
    np.testing.assert_allclose(msgs[4], X[5], rtol=1e-5)

    # Single-neighbor group
    m = run_nmpe_hop(X, [(7, 8)])
    np.testing.assert_allclose(m[7], X[8], rtol=1e-5)

    # VRE batch consistency
    g = vre_gather(X, [2, 3])
    np.testing.assert_allclose(g, np.stack([X[2], X[3]]), rtol=1e-5)

    # Multi-hop slabs: independent per-hop means (as separate DMA transactions)
    slabs = {
        1: SlabHop(1, [(0, 1), (0, 2)]),
        2: SlabHop(2, [(1, 3), (2, 4)]),
    }
    hop1 = run_nmpe_hop(X, slabs[1].edges)
    hop2 = run_nmpe_hop(X, slabs[2].edges)
    np.testing.assert_allclose(hop1[0], (X[1] + X[2]) / 2.0, rtol=1e-5)
    assert 1 in hop2 and 2 in hop2

    print("nmpe_reference: all self-tests passed.")


if __name__ == "__main__":
    _self_test()
