// In-situ aggregation unit — per-seed mean over a stream of neighbor feature vectors.
// s_group_last marks the final neighbor in the current seed's hop group.
module in_situ_aggregation_unit #(
    parameter int unsigned FEAT_DIM = 16,
    parameter int unsigned ACC_W = 32,
    parameter int unsigned CNT_W = 16
) (
    input  logic clk,
    input  logic rst_n,

    input  logic                     s_valid,
    output logic                     s_ready,
    input  logic [FEAT_DIM*ACC_W-1:0] s_feat,
    input  logic                     s_group_last,

    output logic                     m_valid,
    input  logic                     m_ready,
    output logic [FEAT_DIM*ACC_W-1:0] m_mean_feat
);

  typedef enum logic [1:0] { ST_IDLE, ST_RUN, ST_EMIT } state_t;
  state_t state;

  logic [ACC_W-1:0] acc [FEAT_DIM];
  logic [CNT_W-1:0] cnt;
  integer i, j;

  // Hold inputs while completing handshake on aggregated output
  assign s_ready = (state != ST_EMIT) && (!m_valid || m_ready);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state   <= ST_IDLE;
      m_valid <= 1'b0;
      cnt     <= '0;
      for (i = 0; i < FEAT_DIM; i = i + 1)
        acc[i] <= '0;
    end else begin
      if (m_valid && m_ready)
        m_valid <= 1'b0;

      case (state)
        ST_IDLE: begin
          if (s_valid && s_ready) begin
            cnt <= 1;
            for (j = 0; j < FEAT_DIM; j = j + 1)
              acc[j] <= s_feat[j*ACC_W +: ACC_W];
            if (s_group_last)
              state <= ST_EMIT;
            else
              state <= ST_RUN;
          end
        end
        ST_RUN: begin
          if (s_valid && s_ready) begin
            cnt <= cnt + 1'b1;
            for (j = 0; j < FEAT_DIM; j = j + 1)
              acc[j] <= acc[j] + s_feat[j*ACC_W +: ACC_W];
            if (s_group_last)
              state <= ST_EMIT;
          end
        end
        ST_EMIT: begin
          if (!m_valid) begin
            for (j = 0; j < FEAT_DIM; j = j + 1)
              m_mean_feat[j*ACC_W +: ACC_W] <= acc[j] / cnt;
            m_valid <= 1'b1;
          end else if (m_ready) begin
            m_valid <= 1'b0;
            state <= ST_IDLE;
          end
        end
        default: state <= ST_IDLE;
      endcase
    end
  end

endmodule
