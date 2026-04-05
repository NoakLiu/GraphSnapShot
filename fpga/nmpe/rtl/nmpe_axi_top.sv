// NMPE datapath top: SEMHS slab AXI read + edge stream (connect HLS VRE + IAU externally).
module nmpe_axi_top #(
    parameter int unsigned ADDR_W = 64,
    parameter int unsigned AXI_DATA_W = 64,
    parameter int unsigned AXI_ID_W = 4,
    parameter int unsigned FEAT_DIM = 16
) (
    input  logic aclk,
    input  logic aresetn,

    input  logic                     dma_cmd_valid,
    output logic                     dma_cmd_ready,
    input  logic [ADDR_W-1:0]        dma_slab_base,
    input  logic [31:0]              dma_byte_len,
    input  logic [7:0]               dma_hop_id,

    output logic                     edge_valid,
    input  logic                     edge_ready,
    output logic [31:0]              edge_src,
    output logic [31:0]              edge_dst,
    output logic                     edge_last,

    output logic [ADDR_W-1:0]        m_axi_araddr,
    output logic [7:0]               m_axi_arlen,
    output logic [2:0]               m_axi_arsize,
    output logic [1:0]               m_axi_arburst,
    output logic [AXI_ID_W-1:0]      m_axi_arid,
    output logic                     m_axi_arvalid,
    input  logic                     m_axi_arready,
    input  logic [AXI_DATA_W-1:0]    m_axi_rdata,
    input  logic [1:0]               m_axi_rresp,
    input  logic                     m_axi_rlast,
    input  logic [AXI_ID_W-1:0]      m_axi_rid,
    input  logic                     m_axi_rvalid,
    output logic                     m_axi_rready,

    input  logic                     feat_valid,
    output logic                     feat_ready,
    input  logic [FEAT_DIM*32-1:0]   feat_vec,
    input  logic                     feat_group_last,

    output logic                     agg_valid,
    input  logic                     agg_ready,
    output logic [FEAT_DIM*32-1:0]  agg_mean_vec,

    output logic                     dma_busy,
    output logic                     dma_err
);

  slab_dma_controller #(
      .ADDR_W(ADDR_W),
      .AXI_DATA_W(AXI_DATA_W),
      .AXI_ID_W(AXI_ID_W)
  ) u_slab (
      .aclk(aclk),
      .aresetn(aresetn),
      .cmd_valid(dma_cmd_valid),
      .cmd_ready(dma_cmd_ready),
      .slab_base_addr(dma_slab_base),
      .byte_length(dma_byte_len),
      .hop_id(dma_hop_id),
      .m_valid(edge_valid),
      .m_ready(edge_ready),
      .m_src(edge_src),
      .m_dst(edge_dst),
      .m_last_edge(edge_last),
      .m_axi_araddr(m_axi_araddr),
      .m_axi_arlen(m_axi_arlen),
      .m_axi_arsize(m_axi_arsize),
      .m_axi_arburst(m_axi_arburst),
      .m_axi_arid(m_axi_arid),
      .m_axi_arvalid(m_axi_arvalid),
      .m_axi_arready(m_axi_arready),
      .m_axi_rdata(m_axi_rdata),
      .m_axi_rresp(m_axi_rresp),
      .m_axi_rlast(m_axi_rlast),
      .m_axi_rid(m_axi_rid),
      .m_axi_rvalid(m_axi_rvalid),
      .m_axi_rready(m_axi_rready),
      .busy(dma_busy),
      .err(dma_err)
  );

  in_situ_aggregation_unit #(
      .FEAT_DIM(FEAT_DIM),
      .ACC_W(32),
      .CNT_W(16)
  ) u_iau (
      .clk(aclk),
      .rst_n(aresetn),
      .s_valid(feat_valid),
      .s_ready(feat_ready),
      .s_feat(feat_vec),
      .s_group_last(feat_group_last),
      .m_valid(agg_valid),
      .m_ready(agg_ready),
      .m_mean_feat(agg_mean_vec)
  );

endmodule
