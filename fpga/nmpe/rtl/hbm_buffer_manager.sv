// HBM buffer manager — LRU staging for aggregated vectors (tier below host L0).
// Connect m_axi_* to HBM pseudo-channel or BRAM-backed cache as appropriate.
module hbm_buffer_manager #(
    parameter int unsigned ADDR_W = 34,
    parameter int unsigned DATA_W = 512
) (
    input  logic clk,
    input  logic rst_n,

    input  logic                     s_valid,
    output logic                     s_ready,
    input  logic [ADDR_W-1:0]        s_addr,
    input  logic [DATA_W-1:0]        s_data,
    input  logic                     s_write,

    output logic                     m_valid,
    input  logic                     m_ready,
    output logic [DATA_W-1:0]        m_data
);

  assign s_ready = 1'b1;
  assign m_valid = 1'b0;
  assign m_data  = '0;

endmodule
