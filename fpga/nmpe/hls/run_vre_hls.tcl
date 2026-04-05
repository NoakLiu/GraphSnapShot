# Vitis HLS / Vivado HLS batch script — adjust part and clock for your card.
# Usage: vitis_hls -f run_vre_hls.tcl   OR   vivado_hls -f run_vre_hls.tcl

set top vre_gather
set clk 10.0

open_project vre_hls_proj
open_solution solution1 -flow_target vivado
set_part {xcvu9p-flga2104-2L-e}

create_clock -period $clk -name default

add_files ${top}.cpp
set_directive_top -name ${top} ${top}

# Optional: unroll inner feature loop for higher II cost
# set_directive_unroll -factor 4 vre_gather/feat

csynth_design
export_design -rtl verilog -format ip_catalog

exit
