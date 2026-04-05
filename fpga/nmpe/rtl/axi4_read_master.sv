// AXI4 read-only master (INCR). Multi-burst; AXI-Stream output with proper R backpressure.
module axi4_read_master #(
    parameter int unsigned ADDR_W = 64,
    parameter int unsigned DATA_W = 64,
    parameter int unsigned ID_W = 4,
    parameter int unsigned MAX_AXI_LEN = 255
) (
    input  logic aclk,
    input  logic aresetn,

    input  logic                     i_start,
    input  logic [ADDR_W-1:0]        i_start_addr,
    input  logic [31:0]              i_byte_len,
    output logic                     o_idle,

    output logic [ADDR_W-1:0]        m_axi_araddr,
    output logic [7:0]               m_axi_arlen,
    output logic [2:0]               m_axi_arsize,
    output logic [1:0]               m_axi_arburst,
    output logic [ID_W-1:0]          m_axi_arid,
    output logic                     m_axi_arvalid,
    input  logic                     m_axi_arready,

    input  logic [DATA_W-1:0]        m_axi_rdata,
    input  logic [1:0]               m_axi_rresp,
    input  logic                     m_axi_rlast,
    input  logic [ID_W-1:0]          m_axi_rid,
    input  logic                     m_axi_rvalid,
    output logic                     m_axi_rready,

    output logic [DATA_W-1:0]        o_tdata,
    output logic                     o_tvalid,
    input  logic                     o_tready,
    output logic                     o_tlast,
    output logic                     o_err
);

  localparam int unsigned BYTES_PER_BEAT = DATA_W / 8;
  localparam logic [2:0] ARSIZE_VAL = $clog2(BYTES_PER_BEAT)[2:0];

  typedef enum logic [1:0] { ST_IDLE, ST_ISSUE_AR, ST_READ } state_t;
  state_t state;

  logic [ADDR_W-1:0] ar_addr;
  logic [31:0]       remain_bytes;
  logic [8:0]        beats_this_burst;
  logic [8:0]        beat_got;

  assign o_idle = (state == ST_IDLE);
  assign m_axi_rready = (state == ST_READ) && (!o_tvalid || o_tready);

  function automatic logic [8:0] beats_for_bytes(input logic [31:0] b);
    logic [31:0] q;
    q = (b + BYTES_PER_BEAT - 1) / BYTES_PER_BEAT;
    if (q > (MAX_AXI_LEN + 1))
      beats_for_bytes = MAX_AXI_LEN + 1;
    else
      beats_for_bytes = q[8:0];
  endfunction

  logic [8:0] nb_issue;
  always_comb nb_issue = beats_for_bytes(remain_bytes);

  always_ff @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
      state            <= ST_IDLE;
      m_axi_arvalid    <= 1'b0;
      m_axi_araddr     <= '0;
      m_axi_arlen      <= '0;
      m_axi_arid       <= '0;
      ar_addr          <= '0;
      remain_bytes     <= '0;
      beats_this_burst <= '0;
      beat_got         <= '0;
      o_tvalid         <= 1'b0;
      o_tlast          <= 1'b0;
      o_tdata          <= '0;
      o_err            <= 1'b0;
    end else begin
      if (o_tvalid && o_tready) begin
        o_tvalid <= 1'b0;
        o_tlast  <= 1'b0;
      end

      case (state)
        ST_IDLE: begin
          m_axi_arvalid <= 1'b0;
          if (i_start && i_byte_len != 0) begin
            ar_addr      <= i_start_addr;
            remain_bytes <= i_byte_len;
            state        <= ST_ISSUE_AR;
            o_err        <= 1'b0;
          end
        end

        ST_ISSUE_AR: begin
          beats_this_burst <= nb_issue;
          beat_got         <= 9'd0;
          m_axi_araddr     <= ar_addr;
          m_axi_arlen      <= nb_issue[7:0] - 8'd1;
          m_axi_arsize     <= ARSIZE_VAL;
          m_axi_arburst    <= 2'b01;
          m_axi_arid       <= '0;
          m_axi_arvalid    <= 1'b1;
          if (m_axi_arready) begin
            m_axi_arvalid <= 1'b0;
            state         <= ST_READ;
          end
        end

        ST_READ: begin
          if (m_axi_rvalid && m_axi_rready) begin
            if (m_axi_rresp != 2'b00)
              o_err <= 1'b1;

            o_tdata  <= m_axi_rdata;
            o_tvalid <= 1'b1;
            o_tlast  <= m_axi_rlast && (remain_bytes <= BYTES_PER_BEAT);

            if (remain_bytes > BYTES_PER_BEAT)
              remain_bytes <= remain_bytes - BYTES_PER_BEAT;
            else
              remain_bytes <= 0;

            beat_got <= beat_got + 1'b1;

            if (m_axi_rlast) begin
              ar_addr <= ar_addr + beats_this_burst * BYTES_PER_BEAT;
              if (remain_bytes <= BYTES_PER_BEAT) begin
                state <= ST_IDLE;
              end else begin
                state <= ST_ISSUE_AR;
              end
            end
          end
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

endmodule
