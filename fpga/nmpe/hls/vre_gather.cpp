/**
 * Vitis HLS — NMPE Vector Retrieval Engine (VRE).
 *
 * Row-major X with stride in bytes (matches DGL `g.ndata['feat']` contiguous layout).
 * Processes num_lookup consecutive AXI-Stream neighbor IDs per kernel start.
 */
#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include <stdint.h>
#include <string.h>

#ifndef VRE_FEAT_DIM
#define VRE_FEAT_DIM 128
#endif

#ifndef VRE_AXIS_FEAT_WIDTH
#define VRE_AXIS_FEAT_WIDTH (VRE_FEAT_DIM * 32)
#endif

typedef ap_axiu<32, 0, 0, 0> nmpe_id_pkt_t;
typedef ap_axiu<VRE_AXIS_FEAT_WIDTH, 0, 0, 0> nmpe_feat_pkt_t;

void vre_gather(
    hls::stream<nmpe_id_pkt_t> &s_neighbor_id,
    hls::stream<nmpe_feat_pkt_t> &m_feat_vec,
    const float *feature_table,
    uint32_t num_nodes,
    uint32_t feat_dim,
    uint32_t feat_stride_bytes,
    uint32_t num_lookup) {
#pragma HLS INTERFACE mode=axis port=s_neighbor_id
#pragma HLS INTERFACE mode=axis port=m_feat_vec
#pragma HLS INTERFACE mode=m_axi bundle=GMEM0 port=feature_table offset=slave latency=64 \
    num_read_outstanding=32 max_read_burst_length=256
#pragma HLS INTERFACE mode=s_axilite port=num_nodes bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=feat_dim bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=feat_stride_bytes bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=num_lookup bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=return bundle=CTRL

  uint32_t dlim = feat_dim;
  if (dlim > VRE_FEAT_DIM)
    dlim = VRE_FEAT_DIM;

  uint32_t stride_elems = feat_stride_bytes / sizeof(float);
  if (stride_elems == 0)
    stride_elems = dlim;

lookup:
  for (uint32_t i = 0; i < num_lookup; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 65536 avg = 1024
    nmpe_id_pkt_t id_pkt = s_neighbor_id.read();
    uint32_t nid = id_pkt.data;

    ap_uint<VRE_AXIS_FEAT_WIDTH> packed;
    uint64_t base = (uint64_t)nid * (uint64_t)stride_elems;

    if (nid < num_nodes) {
    feat:
      for (uint32_t d = 0; d < dlim; d++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 16 max = VRE_FEAT_DIM avg = 100
        float v = feature_table[base + d];
        uint32_t bits;
        memcpy(&bits, &v, sizeof(bits));
        packed((d + 1) * 32 - 1, d * 32) = bits;
      }
    } else {
      packed = 0;
    }

    nmpe_feat_pkt_t out;
    out.data = packed;
    out.keep = -1;
    out.strb = -1;
    out.last = id_pkt.last;
    out.dest = 0;
    out.user = 0;
    m_feat_vec.write(out);
  }
}
