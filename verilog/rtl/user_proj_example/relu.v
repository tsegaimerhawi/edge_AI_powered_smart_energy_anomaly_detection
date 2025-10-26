/*
  relu activation function
*/

`timescale 1ns / 1ps

module relu #
(
    parameter N = 32
)
(
    input  wire [N-1:0] relu_i,
    output wire [N-1:0] relu_o
);
    assign relu_o = (relu_i[31] == 0) ? relu_i : 0;

endmodule

