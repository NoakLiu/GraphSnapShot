# NMPE hardware ABI: AXI4, HLS VRE, DGL-aligned descriptors

This document ties together **real AXI**, the **Vitis HLS `vre_gather`** kernel, and the **host job descriptor** layout used next to a DGL `DataLoader` loop.

## 1. PCIe BAR0 / MMIO (`graphsnap_nmpe_descriptors.h`)

| Offset | Name | Notes |
|--------|------|--------|
| 0x00 | CTRL | START, SOFT_RST, IRQ_EN |
| 0x04 | STATUS | BUSY, ERR |
| 0x08 | SQ_TAIL | Producer index (host writes after enqueue) |
| 0x0C | CQ_HEAD | Consumer index |
| 0x10–0x1C | SQ_BASE_LO/HI, CQ_BASE_LO/HI | Physical addresses of pinned queues |
| 0x20–0x24 | SQ_DEPTH, CQ_DEPTH | Number of slots |
| 0x28 | VERSION | `(major<<16)|minor` — use `NMPE_ABI_VERSION` |

Submission queue entries are **`nmpe_job_desc_t`** (92 bytes, packed). The Python mirror is `fpga/nmpe/host/nmpe_descriptors.py`.

## 2. Aligning with DGL `NeighborSampler` / minibatch

DGL does not define FPGA packets. Map your training step as follows:

1. **Sampler (CPU or GPU)**  
   Use `dgl.dataloading.NeighborSampler(fanouts)` (or GraphSnapShot SSDReS samplers) to obtain **blocks** and node IDs per layer.

2. **Pin host buffers**  
   - `feature_table_addr`: base of `g.ndata['feat'].float().contiguous()` (or UVA policy you use with `use_uva`).  
   - `feat_dim`, `feat_stride_bytes`: `feat.shape[1] * 4` when row-major contiguous.  
   - `fanout_this_hop` / `hop_index`: same semantics as `fanouts[hop_index]` for the slab you offload.  
   - `batch_seed_count`: number of seed nodes in the minibatch (e.g. `batch_size`).  
   - `seed_ids_addr`: `output_nodes` (or layer seeds) as `uint32_t` array.  
   - `group_count_addr`: per-seed neighbor count (`uint16_t` or `uint32_t` — firmware must match; header documents 16-bit).  
   - `neighbor_ids_addr`: CSR-style flat list of sampled neighbor IDs (order matches `group_count`).  
   - `slab.dev_addr` / `num_edges`: SEMHS slab blob for that hop (pairs `src,dst` as two little-endian `uint32` per 8 bytes).

3. **Enqueue**  
   Write one `nmpe_job_desc_t` per SQ slot, advance **SQ_TAIL**, pulse **CTRL.START** (or doorbell your RTL implements).

4. **Completion**  
   Poll **CQ** or MSI; `nmpe_cqe_t` echoes `user_cookie`.

**Important:** The descriptor’s `group_count_addr` is documented as `uint16_t` in the header comment but the struct only carries pointers — **choose element width in firmware and match it in Python** when packing side tables.

## 3. AXI4 read master (`rtl/axi4_read_master.sv`)

- INCR bursts, `arsize` = log2(bytes per beat).  
- Default **64-bit data** so each beat is one `(src,dst)` edge (LE: `rdata[31:0]=src`, `rdata[63:32]=dst`).  
- `m_axi_rready` = `(!o_tvalid || o_tready)` so the read channel respects AXI-Stream backpressure.

## 4. Slab DMA wrapper (`rtl/slab_dma_controller.sv`)

Connects `cmd_*` to pulse the read master with `slab_base_addr` and `byte_length` (must equal `num_edges * 8` for packed edges). Exposes full **M_AXI** read port to hang on SmartConnect / HBM / DDR.

## 5. HLS VRE (`hls/vre_gather.cpp`)

- **AXIS in:** `ap_axiu<32,...>` neighbor IDs (`TLAST` per logical group if you chain to IAU).  
- **AXIS out:** one packed feature vector per id, width `VRE_FEAT_DIM * 32` (default 128 floats).  
- **M_AXI:** `feature_table` row-major with `feat_stride_bytes` (matches descriptor / DGL layout).  
- **S_AXILITE:** `num_nodes`, `feat_dim`, `feat_stride_bytes`, `num_lookup`.

Synthesis example:

```bash
cd fpga/nmpe/hls
vitis_hls -f run_vre_hls.tcl
```

Change `set_part` in `run_vre_hls.tcl` for Alveo U280 (`xcu280p` variant) or your device.

## 6. Integration into Vivado

1. Add **AXI Interconnect** (or SmartConnect): slave port to HLS VRE `GMEM0`, same address space as host-visible feature table.  
2. Instantiate **`slab_dma_controller`** M_AXI on another master port (or shared if coherent).  
3. **AXIS** connect: slab unpack `m_dst` (neighbor id) → `vre_gather` `s_neighbor_id`; `m_feat_vec` → `in_situ_aggregation_unit` (may need width adapter if `FEAT_DIM` differs).  
4. Export **PCIe** BAR decode to drive `SQ_TAIL` / job fetch FSM (soft CPU or RTL dispatcher).

## 7. Python descriptor sanity check

```bash
python3 fpga/nmpe/host/nmpe_descriptors.py
```

Expect `packed job_desc bytes: 92`.
