module argmax_csw #(
    parameter N  = 4,
    parameter W  = 16,
    parameter Y_AW = 2
)(
    input  wire clk,
    input  wire rst_n,          // active-low
    input  wire start,

    // Y RAM (active-HIGH CS/WE, registered read)
    output reg               y_cs,
    output reg               y_we,
    output reg [Y_AW-1:0]    y_addr,
    output reg [W-1:0]       y_din,
    input  wire [W-1:0]      y_dout,

    output reg               done,
    output reg [$clog2(N)-1:0] index,
    output reg signed [W-1:0]  maxval
);
    localparam signed [W-1:0] VERY_NEG = {1'b1,{W-1{1'b0}}};
    reg signed [W-1:0] cur;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_cs<=0; y_we<=0; y_addr<='0; y_din<='0;
            done<=0; index<='0; maxval<=VERY_NEG; cur<='0;
        end else begin
            done<=1'b0; y_we<=1'b0; y_cs<=1'b0;
            if (start) begin
                index<='0; maxval<=VERY_NEG;
                y_addr<='0; y_cs<=1'b1; // first read issued
            end else begin
                y_cs <= 1'b1;
                cur  <= y_dout;         // data from previous address
                if (cur > maxval) begin
                    maxval <= cur;
                    index  <= y_addr[$clog2(N)-1:0];
                end
                if (y_addr == N-1) begin
                    done <= 1'b1;
                end else begin
                    y_addr <= y_addr + 1'b1;
                end
            end
        end
    end
endmodule
