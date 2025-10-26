/*
   sram
*/
`timescale 1ns / 1ps
`timescale 1ns/1ps

// 1RW synchronous SRAM: active-HIGH CS/WE, registered read (1-cycle latency).
module sram_1rw #(
    parameter ADDR_W = 8,  // depth = 2^ADDR_W
    parameter DATA_W = 16
)(
    input                   CK,
    input                   CS,
    input                   WE,
    input      [ADDR_W-1:0] A,
    input      [DATA_W-1:0] D,
    output     [DATA_W-1:0] Q
);
    localparam DEPTH = (1<<ADDR_W);
    reg [DATA_W-1:0] mem [0:DEPTH-1];
    reg [DATA_W-1:0] Qr;

    always @(posedge CK) begin
        if (CS) begin
            Qr <= mem[A];           // registered read (visible next cycle)
            if (WE) mem[A] <= D;    // read-before-write
        end
    end
    assign Q = Qr;
endmodule
