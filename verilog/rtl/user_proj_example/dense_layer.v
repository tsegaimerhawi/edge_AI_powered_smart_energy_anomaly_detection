module dense_layer #(
    parameter IN_SIZE   = 7,
    parameter OUT_SIZE  = 32,
    parameter DW        = 16,
    parameter ACC_W     = 40,

    // Address widths (sized for power-of-two depths that cover actual counts)
    parameter X_AW = 3,  // >= ceil(log2(IN_SIZE))
    parameter W_AW = 8,  // >= ceil(log2(IN_SIZE*OUT_SIZE))
    parameter B_AW = 5,  // >= ceil(log2(OUT_SIZE))
    parameter Y_AW = 5   // >= ceil(log2(OUT_SIZE))
)(
    input  wire clk,
    input  wire rst_n,         // active-low
    input  wire start,
    input  wire [1:0] act_sel, // 0=linear, 1=ReLU
    output reg  done,

    // X RAM
    output reg               x_cs,
    output reg               x_we,
    output reg [X_AW-1:0]    x_addr,
    output reg [DW-1:0]      x_din,
    input  wire [DW-1:0]     x_dout,

    // W RAM (flattened: addr = n*IN_SIZE + i)
    output reg               w_cs,
    output reg               w_we,
    output reg [W_AW-1:0]    w_addr,
    output reg [DW-1:0]      w_din,
    input  wire [DW-1:0]     w_dout,

    // B RAM (bias per neuron)
    output reg               b_cs,
    output reg               b_we,
    output reg [B_AW-1:0]    b_addr,
    output reg [DW-1:0]      b_din,
    input  wire [DW-1:0]     b_dout,

    // Y RAM
    output reg               y_cs,
    output reg               y_we,
    output reg [Y_AW-1:0]    y_addr,
    output reg [DW-1:0]      y_din,
    input  wire [DW-1:0]     y_dout
);
    localparam S_IDLE  = 0;
    localparam S_BIAS  = 1;
    localparam S_PRIME = 2;
    localparam S_MAC   = 3;
    localparam S_WRITE = 4;
    localparam S_NEXT  = 5;

    reg [2:0] state;

    reg [$clog2(IN_SIZE)-1:0]  i;   // input index
    reg [$clog2(OUT_SIZE)-1:0] n;   // neuron index

    // MAC core connections
    wire signed [DW-1:0] x_s = x_dout;
    wire signed [DW-1:0] w_s = w_dout;
    wire signed [DW-1:0] y_mac;
    wire mac_done;

    // Bias sign-extend into accumulator domain
    wire signed [ACC_W-1:0] bias_acc;

    neuron_core #(
        .N(16), .ACC_WIDTH(40)
    ) mac (
        .clk(clk), .rst_n(rst_n),
        .start(state==S_BIAS),
        .act_sel(act_sel), .bias_acc(bias_acc),
        .data_i(x_s), .weight_i(w_s),
        .xw_val(state==S_MAC),
        .xw_last (state==S_MAC && (i==IN_SIZE-1)),
        .out_o(y_mac), .done(mac_done)
    );

    wire [W_AW-1:0] w_base = n*IN_SIZE;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done <= 1'b0; i<='0; n<='0;
            x_cs<=0; x_we<=0; x_addr<='0; x_din<='0;
            w_cs<=0; w_we<=0; w_addr<='0; w_din<='0;
            b_cs<=0; b_we<=0; b_addr<='0; b_din<='0;
            y_cs<=0; y_we<=0; y_addr<='0; y_din<='0;
        end else begin
            done <= 1'b0;
            x_cs<=0; x_we<=0;
            w_cs<=0; w_we<=0;
            b_cs<=0; b_we<=0;
            y_cs<=0; y_we<=0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        n      <= '0;
                        i      <= '0;
                        b_addr <= '0;
                        state  <= S_BIAS;
                    end
                end

                // Issue bias read + first X/W addresses (data valid next cycle)
                S_BIAS: begin
                    b_cs   <= 1'b1;
                    x_cs   <= 1'b1; x_addr <= '0;
                    w_cs   <= 1'b1; w_addr <= w_base;
                    state  <= S_PRIME;
                end

                // First valid cycle for X/W/B outputs on Q ports
                S_PRIME: begin
                    x_cs <= 1'b1; w_cs <= 1'b1; b_cs <= 1'b1;
                    i    <= '0;
                    state<= S_MAC;
                end

                // Stream IN_SIZE pairs into MAC
                S_MAC: begin
                    x_cs <= 1'b1; w_cs <= 1'b1; b_cs <= 1'b1;
                    if (i < IN_SIZE-1) begin
                        i      <= i + 1'b1;
                        x_addr <= x_addr + 1'b1;
                        w_addr <= w_addr + 1'b1;
                    end else begin
                        state  <= S_WRITE;
                    end
                end

                // Write Y[n] once MAC is done
                S_WRITE: begin
                    if (mac_done) begin
                        y_cs   <= 1'b1;
                        y_we   <= 1'b1;
                        y_addr <= n;
                        y_din  <= y_mac;
                        state  <= S_NEXT;
                    end
                end

                // Next neuron or finish layer
                S_NEXT: begin
                    if (n < OUT_SIZE-1) begin
                        n      <= n + 1'b1;
                        b_addr <= n + 1'b1;
                        state  <= S_BIAS;
                    end else begin
                        done   <= 1'b1;
                        state  <= S_IDLE;
                    end
                end
            endcase
        end
    end
endmodule
