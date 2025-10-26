/*
   neuron_core
*/
`timescale 1ns / 1ps

module neuron_core #
(
   parameter N=16,
   parameter ACC_WIDTH=40
)
(
   input  wire                        clk,
   input  wire                        rst_n,
   input  wire                        start,
   input  wire        [          1:0] act_sel, //0-linear, 1= relu

   input  wire signed [        N-1:0] data_i,
   input  wire signed [        N-1:0] weight_i,
   input  wire                        xw_val,
   input  wire                        xw_last,
   input  wire signed [ACC_WIDTH-1:0] bias_acc,

   output reg  signed [        N-1:0] out_o,
   output reg                         done
);

   reg  signed [ACC_WIDTH-1:0] acc;
   wire signed [(2*N)-1:0] prod;

   // On start, load bias; otherwise keep acc and add product if valid
   wire signed [ACC_WIDTH-1:0] acc_next;

   // Optional ReLU (no sat/round)
   wire signed [ACC_WIDTH-1:0] acc_act;

   always @(posedge clk or negedge rst_n) begin
      if (~rst_n) begin
         acc  <= '0;
         out_o    <= '0;
         done <= 1'b0;
      end else begin
         done <= 1'b0;
         acc  <= acc_next;
         if (xw_val && xw_last) begin
               out_o <= acc_act[N-1:0];
               done  <= 1'b1;
         end
      end
   end

   assign prod = data_i * weight_i;
   assign acc_next = (start ? bias_acc : acc) + (xw_val ? prod : '0);
   assign acc_act = (act_sel==2'd1 && acc_next[ACC_WIDTH-1]) ? {ACC_WIDTH{1'b0}} : acc_next;

endmodule
