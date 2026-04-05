// GraphSnapShot NMPE — shared parameters and types
package nmpe_pkg;

  // Neighbor-ID / seed-ID width (matches host descriptor layout)
  parameter int unsigned NODE_ID_W = 32;

  // Feature vector elements (paper uses d=128 for throughput estimates)
  parameter int unsigned FEAT_DIM_DEFAULT = 128;
  parameter int unsigned FEAT_ELEM_W = 32;

  // VRE lane parallelism (paper: P=512 for U280-style estimates)
  parameter int unsigned P_LANES_DEFAULT = 512;

  // AXI4 burst alignment: SEMHS slab granularity in 512-bit LBA units (platform-specific)
  parameter int unsigned SLAB_ALIGN_SHIFT = 6;

endpackage
