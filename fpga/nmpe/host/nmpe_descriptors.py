"""
Pack/unpack NMPE job descriptors shared with FPGA BAR + pinned memory.

Maps naturally to a DGL minibatch when you already have:
  - seed node ids (batch output nodes)
  - per-seed neighbor counts and flat neighbor id list (from sample_neighbors / SEMHS)
  - fanouts[hop] matching NeighborSampler
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Sequence, Tuple

NMPE_MAGIC = 0x4E4D5045  # b'NMPE' LE

NMPE_JOB_F_LAST_IN_BATCH = 1 << 0
NMPE_JOB_F_MEAN_AGG = 1 << 1
NMPE_JOB_F_VALIDATE_IDS = 1 << 2


@dataclass
class SlabRegion:
    dev_addr: int
    num_edges: int
    hop_index: int

    @property
    def byte_length(self) -> int:
        return self.num_edges * 8


@dataclass
class JobDesc:
    flags: int
    feature_table_addr: int
    num_nodes: int
    feat_dim: int
    feat_stride_bytes: int
    fanout_this_hop: int
    batch_seed_count: int
    seed_ids_addr: int
    group_count_addr: int
    neighbor_ids_addr: int
    out_feat_addr: int
    slab: SlabRegion
    user_cookie: int = 0

    def pack(self) -> bytes:
        """Binary layout must match nmpe_job_desc_t in graphsnap_nmpe_descriptors.h."""
        slab = self.slab
        # Order matches nmpe_job_desc_t in graphsnap_nmpe_descriptors.h (92 bytes).
        return struct.pack(
            "<2IQ5I4QQIIII",
            NMPE_MAGIC,
            self.flags,
            self.feature_table_addr,
            self.num_nodes,
            self.feat_dim,
            self.feat_stride_bytes,
            self.fanout_this_hop,
            self.batch_seed_count,
            self.seed_ids_addr,
            self.group_count_addr,
            self.neighbor_ids_addr,
            self.out_feat_addr,
            slab.dev_addr,
            slab.num_edges,
            slab.hop_index,
            0,
            self.user_cookie,
        )


@dataclass
class Completion:
    user_cookie: int
    status: int
    cycles: int

    @staticmethod
    def unpack(blob: bytes) -> "Completion":
        user_cookie, status, cycles = struct.unpack("<IIQ", blob)
        return Completion(user_cookie, status, cycles)


def feat_stride(feat_dim: int, elem_bytes: int = 4) -> int:
    return feat_dim * elem_bytes


def example_from_fanouts(
    fanouts: Sequence[int],
) -> Tuple[JobDesc, dict]:
    """
    Illustrate field correspondence with dgl.dataloading.NeighborSampler(fanouts).

    After one batch you would set:
      batch_seed_count = number of seed nodes in the batch
      fanout_this_hop = fanouts[hop_index] for the slab you are offloading
    """
    hop_index = 0
    desc = JobDesc(
        flags=NMPE_JOB_F_MEAN_AGG,
        feature_table_addr=0x1000_0000,
        num_nodes=2_449_029,
        feat_dim=100,
        feat_stride_bytes=400,
        fanout_this_hop=fanouts[hop_index],
        batch_seed_count=1024,
        seed_ids_addr=0x2000_0000,
        group_count_addr=0x2000_1000,
        neighbor_ids_addr=0x2000_2000,
        out_feat_addr=0x2000_8000,
        slab=SlabRegion(dev_addr=0x3000_0000, num_edges=50_000, hop_index=hop_index),
        user_cookie=1,
    )
    meta = {
        "neighbor_sampler_fanouts": list(fanouts),
        "hop_index": hop_index,
    }
    return desc, meta


if __name__ == "__main__":
    d, _ = example_from_fanouts([15, 10, 5])
    b = d.pack()
    assert len(b) == 92, len(b)
    print("packed job_desc bytes:", len(b))
