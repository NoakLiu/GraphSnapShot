// SEMHS slab DMA: wraps axi4_read_master; each AXI beat is one edge (LE: low32=src, high32=dst).
module slab_dma_controller #(
    parameter int unsigned ADDR_W = 64,
    parameter int unsigned AXI_DATA_W = 64,
    parameter int unsigned AXI_ID_W = 4
) (
    input  logic aclk,
    input  logic aresetn,

    input  logic                     cmd_valid,
    output logic                     cmd_ready,
    input  logic [ADDR_W-1:0]        slab_base_addr,
    input  logic [31:0]              byte_length,
    input  logic [7:0]               hop_id,

    output logic                     m_valid,
    input  logic                     m_ready,
    output logic [31:0]              m_src,
    output logic [31:0]              m_dst,
    output logic                     m_last_edge,

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

    output logic                     busy,
    output logic                     err
);

  logic        dma_pulse;
  logic [ADDR_W-1:0] latched_addr;
  logic [31:0]       latched_len;
  logic              xfer_active;

  logic [AXI_DATA_W-1:0] tdata;
  logic                  tvalid;
  logic                  tlast;
  logic                  dma_idle;

  assign m_src       = tdata[31:0];
  assign m_dst       = tdata[63:32];
  assign m_valid     = tvalid;
  assign m_last_edge = tvalid && tlast;
  assign busy        = xfer_active || (cmd_valid && !cmd_ready);

  axi4_read_master #(
      .ADDR_W(ADDR_W),
      .DATA_W(AXI_DATA_W),
      .ID_W(AXI_ID_W)
  ) u_axi (
      .aclk(aclk),
      .aresetn(aresetn),
      .i_start(dma_pulse),
      .i_start_addr(latched_addr),
      .i_byte_len(latched_len),
      .o_idle(dma_idle),
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
      .o_tdata(tdata),
      .o_tvalid(tvalid),
      .o_tready(m_ready),
      .o_tlast(tlast),
      .o_err(err)
  );

  always_ff @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
      dma_pulse    <= 1'b0;
      cmd_ready    <= 1'b1;
      latched_addr <= '0;
      latched_len  <= '0;
      xfer_active  <= 1'b0;
    end else begin
      dma_pulse <= 1'b0;

      if (cmd_valid && cmd_ready && dma_idle && !xfer_active && byte_length != 0) begin
        latched_addr  <= slab_base_addr;
        latched_len   <= byte_length;
        dma_pulse     <= 1'b1;
        cmd_ready     <= 1'b0;
        xfer_active   <= 1'b1;
      end

      if (xfer_active && dma_idle && !dma_pulse) begin
        xfer_active <= 1'b0;
        cmd_ready   <= 1'b1;
      end
    end
  end

endmodule
