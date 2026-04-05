# GraphSnapShot FPGA Near-Memory Processing Engine (NMPE)

Reference RTL + HLS + host ABI for the NMPE in *GraphSnapshot: Acceleration for Graph Vector Retrieval via FPGA Near-Memory Processing* (SEMHS slab DMA → VRE feature gather → IAU mean → HBM / PCIe).

## What is implemented for “real hardware”

| Piece | Location |
|--------|-----------|
| **AXI4 read master** (INCR, burst, stream out, R backpressure) | `rtl/axi4_read_master.sv` |
| **Slab DMA** (command + full `M_AXI` read port + `(src,dst)` stream) | `rtl/slab_dma_controller.sv` |
| **IAU** (streaming mean) | `rtl/in_situ_aggregation_unit.sv` |
| **Top: slab AXI + IAU** (HLS VRE = AXIS insert between `edge_dst` and `feat_vec`) | `rtl/nmpe_axi_top.sv` |
| **Vitis HLS VRE** | `hls/vre_gather.cpp`, `hls/run_vre_hls.tcl` |
| **PCIe/MMIO + job descriptor ABI** (aligned to DGL minibatch fields) | `include/graphsnap_nmpe_descriptors.h`, `host/nmpe_descriptors.py` |
| **Integration / DGL mapping** | `docs/HARDWARE.md` |
| **Python golden model** | `sim/nmpe_reference.py` |

## Architecture (four stages)

1. **Slab DMA** — AXI4 master reads SEMHS slab bytes; each 64-bit beat is LE `src` (low), `dst` (high).
2. **VRE** — HLS `vre_gather`: AXIS `uint32` neighbor id → AXIS packed `feat_dim × fp32` row from `m_axi` feature table (`feat_stride_bytes` like DGL contiguous `ndata['feat']`).
3. **IAU** — `in_situ_aggregation_unit.sv`: mean per seed group (`feat_group_last`).
4. **HBM buffer manager** — still an integration shell (`hbm_buffer_manager.sv`); connect to HBM MC + LRU policy in Vivado.

## Quick checks

```bash
# Descriptor size (must be 92 bytes)
python3 fpga/nmpe/host/nmpe_descriptors.py

# Behavioral pipeline
python3 fpga/nmpe/sim/nmpe_reference.py
```

## HLS

```bash
cd fpga/nmpe/hls
vitis_hls -f run_vre_hls.tcl
```

Edit `set_part` in `run_vre_hls.tcl` for your FPGA.

## Related repo paths

- Samplers: `SSDReS_Samplers/`, `examples/dgl/`
- CUDA: `cuda_kernels/`

## Citation

Cite the GraphSnapShot / GraphSnapshot line of work you use in publications.
