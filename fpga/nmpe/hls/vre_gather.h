/**
 * Vitis HLS top for NMPE Vector Retrieval Engine (VRE).
 * One neighbor id in (AXIS 32-bit) -> feat_dim fp32 out (AXIS 512-bit when dim=16, etc.).
 */
#ifndef VRE_GATHER_H
#define VRE_GATHER_H

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include <stdint.h>

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
    uint32_t num_lookup);

#endif
