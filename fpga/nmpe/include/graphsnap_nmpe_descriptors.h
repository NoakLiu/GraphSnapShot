/**
 * GraphSnapShot NMPE — host/FPGA shared job layout (PCIe BAR + pinned DRAM).
 *
 * DGL does not define FPGA queues. This ABI is the contract between:
 *   - Host: PyTorch/DGL training loop after neighbor sampling (or SEMHS loader)
 *   - Device: NMPE DMA + HLS VRE + IAU
 *
 * All little-endian. Use 64-byte alignment for SQ/CQ bases and slab buffers when
 * programming Alveo/HBM systems.
 */
#ifndef GRAPHSNAP_NMPE_DESCRIPTORS_H
#define GRAPHSNAP_NMPE_DESCRIPTORS_H

#include <stdint.h>

#define NMPE_MAGIC 0x4E4D5045u /* 'NMPE' */

/** CTRL register bits (BAR0 + 0x000) */
#define NMPE_CTRL_SOFT_RST (1u << 0)
#define NMPE_CTRL_START    (1u << 1)
#define NMPE_CTRL_IRQ_EN   (1u << 2)

/** STATUS register bits (BAR0 + 0x004) */
#define NMPE_STS_BUSY      (1u << 0)
#define NMPE_STS_ERR       (1u << 1)
#define NMPE_STS_CQ_OVERFLOW (1u << 2)

/** job_desc.flags */
#define NMPE_JOB_F_LAST_IN_BATCH (1u << 0) /**< raise MSI when done with full minibatch */
#define NMPE_JOB_F_MEAN_AGG      (1u << 1) /**< IAU mean-reduce per seed group */
#define NMPE_JOB_F_VALIDATE_IDS  (1u << 2) /**< clamp / drop OOB node ids in VRE */

/**
 * One SEMHS slab region in coherent memory (host pinned or card DRAM).
 * Edge records are consecutive pairs: uint32_t src, uint32_t dst (row-major).
 */
typedef struct __attribute__((packed)) nmpe_slab_region {
    uint64_t dev_addr; /**< byte address visible to NMPE AXI master */
    uint32_t num_edges; /**< number of (src,dst) pairs; byte_length = num_edges * 8 */
    uint32_t hop_index; /**< 0 = outer hop toward seeds; k-1 = inner — document your hop order */
} nmpe_slab_region_t;

/**
 * Single submission-queue entry. One entry typically corresponds to one hop slab
 * fetch + VRE gather + (optional) IAU for a group of seeds.
 *
 * Alignment with DGL NeighborSampler:
 *   - fanouts[] length == number of GNN layers; hop_index maps to which layer’s
 *     frontier expansion this slab supports (host packs slabs per layer).
 *   - batch_seed_count == |output_nodes| for the minibatch (seeds).
 *   - After sampling, host writes neighbor lists / SEMHS slab bytes, then enqueues
 *     one nmpe_job_desc_t per (hop, slab chunk) or merges slabs — your loader chooses.
 */
typedef struct __attribute__((packed)) nmpe_job_desc {
    uint32_t magic; /**< NMPE_MAGIC */
    uint32_t flags;
    uint64_t feature_table_addr; /**< row-major X[n_nodes][feat_dim] fp32 */
    uint32_t num_nodes;          /**< nrow bound check */
    uint32_t feat_dim;
    uint32_t feat_stride_bytes;  /**< usually feat_dim * 4; allows padding */
    uint32_t fanout_this_hop;    /**< mirrors fanouts[hop_index] in NeighborSampler */
    uint32_t batch_seed_count;
    uint64_t seed_ids_addr;      /**< uint32_t seeds[batch_seed_count] */
    uint64_t group_count_addr;   /**< uint32_t count[batch_seed_count] (use u32 on host; device CSR may narrow) */
    uint64_t neighbor_ids_addr;  /**< flat neighbor ids after sampling (length sum(counts)) */
    uint64_t out_feat_addr;      /**< fp32 [batch_seed_count][feat_dim] aggregated messages */
    nmpe_slab_region_t slab;
    uint32_t reserved0;
    uint32_t user_cookie; /**< echoed in completion */
} nmpe_job_desc_t;

typedef struct __attribute__((packed)) nmpe_cqe {
    uint32_t user_cookie;
    uint32_t status; /**< 0 = OK; else vendor codes */
    uint64_t cycles;
} nmpe_cqe_t;

/** MMIO layout (first 4 KiB of BAR0) — 32-bit word offsets */
enum nmpe_bar0_regs {
    NMPE_REG_CTRL = 0x00 / 4,
    NMPE_REG_STATUS = 0x04 / 4,
    NMPE_REG_SQ_TAIL = 0x08 / 4,
    NMPE_REG_CQ_HEAD = 0x0C / 4,
    NMPE_REG_SQ_BASE_LO = 0x10 / 4,
    NMPE_REG_SQ_BASE_HI = 0x14 / 4,
    NMPE_REG_CQ_BASE_LO = 0x18 / 4,
    NMPE_REG_CQ_BASE_HI = 0x1C / 4,
    NMPE_REG_SQ_DEPTH = 0x20 / 4,
    NMPE_REG_CQ_DEPTH = 0x24 / 4,
    NMPE_REG_VERSION = 0x28 / 4, /**< (major<<16)|minor */
};

#define NMPE_ABI_VERSION ((1u << 16) | 0u)

#endif /* GRAPHSNAP_NMPE_DESCRIPTORS_H */
