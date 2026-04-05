// Legacy minimal top: IAU-only path for unit sim. For AXI slab + IAU use nmpe_axi_top.sv.
module nmpe_top #(
    parameter int unsigned FEAT_DIM = 16,
    parameter int unsigned NODE_ID_W = 32
) (
    input  logic clk,
    input  logic rst_n,

    input  logic                     feat_valid,
    output logic                     feat_ready,
    input  logic [FEAT_DIM*32-1:0]   feat_vec,
    input  logic                     feat_group_last,

    output logic                     agg_valid,
    input  logic                     agg_ready,
    output logic [FEAT_DIM*32-1:0]  agg_mean_vec
);

  in_situ_aggregation_unit #(
      .FEAT_DIM(FEAT_DIM),
      .ACC_W(32),
      .CNT_W(16)
  ) u_iau (
      .clk(clk),
      .rst_n(rst_n),
      .s_valid(feat_valid),
      .s_ready(feat_ready),
      .s_feat(feat_vec),
      .s_group_last(feat_group_last),
      .m_valid(agg_valid),
      .m_ready(agg_ready),
      .m_mean_feat(agg_mean_vec)
  );

endmodule
