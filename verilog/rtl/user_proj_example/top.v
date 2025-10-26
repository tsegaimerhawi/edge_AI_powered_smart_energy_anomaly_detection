`timescale 1ns/1ps

module nn_top_csw_7_32_16_4 #(
    parameter DW        = 16,
    parameter OUT_SHIFT = 0,     // no scaling/shifting as requested
    parameter ACC_W     = 40
)(
    input  wire clk,
    input  wire rst_n,           // active-low
    input  wire start,

    // host write for X0 (7 elements)
    input  wire                  x0_wr_en,
    input  wire [2:0]            x0_wr_addr,    // 0..6
    input  wire signed [DW-1:0]  x0_wr_data,

    output wire                  done,
    output wire [1:0]            class_out,
    output wire signed [DW-1:0]  score_max
);

    // ----------------------------------------------------------------
    // X0 RAM (host OR L1 drives it)  depth 8
    // ----------------------------------------------------------------
    wire [DW-1:0] x0_q;
    reg           x0_cs, x0_we;           // driven by mux (host vs L1)
    reg  [2:0]    x0_a;
    reg  [DW-1:0] x0_d;

    sram_1rw #(.ADDR_W(3), .DATA_W(DW)) X0 (
        .CK(clk), .CS(x0_cs), .WE(x0_we), .A(x0_a), .D(x0_d), .Q(x0_q)
    );

    // ----------------------------------------------------------------
    // L1 memories: W1/B1/Y1
    // ----------------------------------------------------------------
    // W1 32*7=224 -> use 256 depth (8-bit addr)
    wire [DW-1:0] w1_q;  wire w1_cs, w1_we; wire [7:0] w1_a; wire [DW-1:0] w1_d;
    sram_1rw #(.ADDR_W(8), .DATA_W(DW)) W1 (
        .CK(clk), .CS(w1_cs), .WE(w1_we), .A(w1_a), .D(w1_d), .Q(w1_q)
    );

    // B1 32 -> 5-bit addr
    wire [DW-1:0] b1_q;  wire b1_cs, b1_we; wire [4:0] b1_a; wire [DW-1:0] b1_d;
    sram_1rw #(.ADDR_W(5), .DATA_W(DW)) B1 (
        .CK(clk), .CS(b1_cs), .WE(b1_we), .A(b1_a), .D(b1_d), .Q(b1_q)
    );

    // Y1 32 -> 5-bit addr  (shared: writer=L1, reader=L2)
    wire [DW-1:0] y1_q;
    reg           y1_cs, y1_we;           // driven by mux (L1 vs L2)
    reg  [4:0]    y1_a;
    reg  [DW-1:0] y1_d;
    sram_1rw #(.ADDR_W(5), .DATA_W(DW)) Y1 (
        .CK(clk), .CS(y1_cs), .WE(y1_we), .A(y1_a), .D(y1_d), .Q(y1_q)
    );

    // ----------------------------------------------------------------
    // L2 memories: W2/B2/Y2
    // ----------------------------------------------------------------
    // W2 16*32=512 -> 9-bit addr
    wire [DW-1:0] w2_q;  wire w2_cs, w2_we; wire [8:0] w2_a; wire [DW-1:0] w2_d;
    sram_1rw #(.ADDR_W(9), .DATA_W(DW)) W2 (
        .CK(clk), .CS(w2_cs), .WE(w2_we), .A(w2_a), .D(w2_d), .Q(w2_q)
    );

    // B2 16 -> 4-bit addr
    wire [DW-1:0] b2_q;  wire b2_cs, b2_we; wire [3:0] b2_a; wire [DW-1:0] b2_d;
    sram_1rw #(.ADDR_W(4), .DATA_W(DW)) B2 (
        .CK(clk), .CS(b2_cs), .WE(b2_we), .A(b2_a), .D(b2_d), .Q(b2_q)
    );

    // Y2 16 -> 4-bit addr (shared: writer=L2, reader=L3)
    wire [DW-1:0] y2_q;
    reg           y2_cs, y2_we;           // driven by mux (L2 vs L3)
    reg  [3:0]    y2_a;
    reg  [DW-1:0] y2_d;
    sram_1rw #(.ADDR_W(4), .DATA_W(DW)) Y2 (
        .CK(clk), .CS(y2_cs), .WE(y2_we), .A(y2_a), .D(y2_d), .Q(y2_q)
    );

    // ----------------------------------------------------------------
    // L3 memories: W3/B3/Y3
    // ----------------------------------------------------------------
    // W3 4*16=64 -> 6-bit addr
    wire [DW-1:0] w3_q;  wire w3_cs, w3_we; wire [5:0] w3_a; wire [DW-1:0] w3_d;
    sram_1rw #(.ADDR_W(6), .DATA_W(DW)) W3 (
        .CK(clk), .CS(w3_cs), .WE(w3_we), .A(w3_a), .D(w3_d), .Q(w3_q)
    );

    // B3 4 -> 2-bit addr
    wire [DW-1:0] b3_q;  wire b3_cs, b3_we; wire [1:0] b3_a; wire [DW-1:0] b3_d;
    sram_1rw #(.ADDR_W(2), .DATA_W(DW)) B3 (
        .CK(clk), .CS(b3_cs), .WE(b3_we), .A(b3_a), .D(b3_d), .Q(b3_q)
    );

    // Y3 4 -> 2-bit addr (shared: writer=L3, reader=ARG)
    wire [DW-1:0] y3_q;
    reg           y3_cs, y3_we;           // driven by mux (L3 vs ARG)
    reg  [1:0]    y3_a;
    reg  [DW-1:0] y3_d;
    sram_1rw #(.ADDR_W(2), .DATA_W(DW)) Y3 (
        .CK(clk), .CS(y3_cs), .WE(y3_we), .A(y3_a), .D(y3_d), .Q(y3_q)
    );

    // ----------------------------------------------------------------
    // Channel wires from each layer to shared RAMs
    // ----------------------------------------------------------------
    // L1 drives: X0 (reads), W1,B1,Y1(write)
    wire l1_done;
    wire l1_x_cs, l1_x_we; wire [2:0] l1_x_addr;
    wire       w1_cs_w, w1_we_w; wire [7:0] w1_addr_w; wire [DW-1:0] w1_din_w;
    wire       b1_cs_w, b1_we_w; wire [4:0] b1_addr_w; wire [DW-1:0] b1_din_w;
    wire       y1_cs_from_l1, y1_we_from_l1; wire [4:0] y1_addr_from_l1; wire [DW-1:0] y1_din_from_l1;

    // L2 drives: Y1(as X), W2,B2, Y2(write)
    wire l2_done;
    wire x1_cs_from_l2, x1_we_from_l2; wire [4:0] x1_addr_from_l2; // x1_we will be 0
    wire       w2_cs_w, w2_we_w; wire [8:0] w2_addr_w; wire [DW-1:0] w2_din_w;
    wire       b2_cs_w, b2_we_w; wire [3:0] b2_addr_w; wire [DW-1:0] b2_din_w;
    wire       y2_cs_from_l2, y2_we_from_l2; wire [3:0] y2_addr_from_l2; wire [DW-1:0] y2_din_from_l2;

    // L3 drives: Y2(as X), W3,B3, Y3(write)
    wire l3_done;
    wire x2_cs_from_l3, x2_we_from_l3; wire [3:0] x2_addr_from_l3;
    wire       w3_cs_w, w3_we_w; wire [5:0] w3_addr_w; wire [DW-1:0] w3_din_w;
    wire       b3_cs_w, b3_we_w; wire [1:0] b3_addr_w; wire [DW-1:0] b3_din_w;
    wire       y3_cs_from_l3, y3_we_from_l3; wire [1:0] y3_addr_from_l3; wire [DW-1:0] y3_din_from_l3;

    // Argmax drives: Y3(read-only: CS/ADDR)
    wire arg_done;
    wire       y3_cs_from_arg, y3_we_from_arg; wire [1:0] y3_addr_from_arg; wire [DW-1:0] y3_din_from_arg;

    // ----------------------------------------------------------------
    // Layers
    // ----------------------------------------------------------------
    // L1 7→32 (ReLU)
    dense_layer #(
        .IN_SIZE(7), .OUT_SIZE(32),
        .DW(DW), .ACC_W(ACC_W),
        .X_AW(3), .W_AW(8), .B_AW(5), .Y_AW(5)
    ) L1 (
        .clk(clk), .rst_n(rst_n), .start(start), .act_sel(2'd1), .done(l1_done),

        // X0 RAM (read): L1 outputs these (we'll mux with host)
        .x_cs(l1_x_cs), .x_we(l1_x_we), .x_addr(l1_x_addr), .x_din(), .x_dout(x0_q),

        // W1/B1 (read-only): direct wires to RAM
        .w_cs(w1_cs_w), .w_we(w1_we_w), .w_addr(w1_addr_w), .w_din(), .w_dout(w1_q),
        .b_cs(b1_cs_w), .b_we(b1_we_w), .b_addr(b1_addr_w), .b_din(), .b_dout(b1_q),

        // Y1 (write): goes through mux (L1 vs L2)
        .y_cs(y1_cs_from_l1), .y_we(y1_we_from_l1), .y_addr(y1_addr_from_l1), .y_din(y1_din_from_l1), .y_dout(y1_q)
    );

    // Tie child wires to RAM pins (W1/B1 only one driver)
    assign w1_cs = w1_cs_w; assign w1_we = w1_we_w; assign w1_a = w1_addr_w; assign w1_d = w1_din_w;
    assign b1_cs = b1_cs_w; assign b1_we = b1_we_w; assign b1_a = b1_addr_w; assign b1_d = b1_din_w;

    // L2 32→16 (ReLU)
    dense_layer #(
        .IN_SIZE(32), .OUT_SIZE(16),
        .DW(DW), .ACC_W(ACC_W),
        .X_AW(5), .W_AW(9), .B_AW(4), .Y_AW(4)
    ) L2 (
        .clk(clk), .rst_n(rst_n), .start(l1_done), .act_sel(2'd1), .done(l2_done),

        // X from Y1 RAM (read): outputs from L2 (will be muxed onto Y1 bus after L1 done)
        .x_cs(x1_cs_from_l2), .x_we(x1_we_from_l2), .x_addr(x1_addr_from_l2), .x_din(), .x_dout(y1_q),

        // W2/B2 direct
        .w_cs(w2_cs_w), .w_we(w2_we_w), .w_addr(w2_addr_w), .w_din(), .w_dout(w2_q),
        .b_cs(b2_cs_w), .b_we(b2_we_w), .b_addr(b2_addr_w), .b_din(), .b_dout(b2_q),

        // Y2 write (goes to mux Y2)
        .y_cs(y2_cs_from_l2), .y_we(y2_we_from_l2), .y_addr(y2_addr_from_l2), .y_din(y2_din_from_l2), .y_dout(y2_q)
    );

    assign w2_cs = w2_cs_w; assign w2_we = w2_we_w; assign w2_a = w2_addr_w; assign w2_d = w2_din_w;
    assign b2_cs = b2_cs_w; assign b2_we = b2_we_w; assign b2_a = b2_addr_w; assign b2_d = b2_din_w;

    // L3 16→4 (Linear)
    dense_layer #(
        .IN_SIZE(16), .OUT_SIZE(4),
        .DW(DW), .ACC_W(ACC_W),
        .X_AW(4), .W_AW(6), .B_AW(2), .Y_AW(2)
    ) L3 (
        .clk(clk), .rst_n(rst_n), .start(l2_done), .act_sel(2'd0), .done(l3_done),

        // X from Y2 (read): outputs from L3 (will be muxed onto Y2 bus after L2 done)
        .x_cs(x2_cs_from_l3), .x_we(x2_we_from_l3), .x_addr(x2_addr_from_l3), .x_din(), .x_dout(y2_q),

        // W3/B3 direct
        .w_cs(w3_cs_w), .w_we(w3_we_w), .w_addr(w3_addr_w), .w_din(), .w_dout(w3_q),
        .b_cs(b3_cs_w), .b_we(b3_we_w), .b_addr(b3_addr_w), .b_din(), .b_dout(b3_q),

        // Y3 write (goes to mux Y3)
        .y_cs(y3_cs_from_l3), .y_we(y3_we_from_l3), .y_addr(y3_addr_from_l3), .y_din(y3_din_from_l3), .y_dout(y3_q)
    );

    assign w3_cs = w3_cs_w; assign w3_we = w3_we_w; assign w3_a = w3_addr_w; assign w3_d = w3_din_w;
    assign b3_cs = b3_cs_w; assign b3_we = b3_we_w; assign b3_a = b3_addr_w; assign b3_d = b3_din_w;

    // Argmax over Y3
    argmax_csw #(.N(4), .W(DW), .Y_AW(2)) ARG (
        .clk(clk), .rst_n(rst_n), .start(l3_done),
        // drives a separate channel that we mux to Y3 after L3 is done
        .y_cs(y3_cs_from_arg), .y_we(y3_we_from_arg), .y_addr(y3_addr_from_arg), .y_din(y3_din_from_arg), .y_dout(y3_q),
        .done(arg_done), .index(class_out), .maxval(score_max)
    );

    assign done = arg_done;

    // ----------------------------------------------------------------
    // Arbitration / Bus muxing
    // ----------------------------------------------------------------

    // Latch the handover points so the mux stays stable for the consumer
    reg l1_done_seen, l2_done_seen, l3_done_seen;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            l1_done_seen <= 1'b0;
            l2_done_seen <= 1'b0;
            l3_done_seen <= 1'b0;
        end else begin
            if (start)      l1_done_seen <= 1'b0; // new run
            if (l1_done)    l1_done_seen <= 1'b1;
            if (l2_done)    l2_done_seen <= 1'b1;
            if (l3_done)    l3_done_seen <= 1'b1;
        end
    end

    // ---- X0 mux: host writes before start, then L1 owns X0 for reading
    always @(*) begin
        if (x0_wr_en) begin
            x0_cs = 1'b1; x0_we = 1'b1; x0_a = x0_wr_addr; x0_d = x0_wr_data;
        end else begin
            x0_cs = l1_x_cs; x0_we = l1_x_we; x0_a = l1_x_addr; x0_d = '0;
        end
    end

    // ---- Y1 mux: before L1 finishes → L1 writes; after L1 finishes → L2 reads it
    always @(*) begin
        if (!l1_done_seen) begin
            // L1 owns Y1 (writing)
            y1_cs = y1_cs_from_l1; y1_we = y1_we_from_l1; y1_a = y1_addr_from_l1; y1_d = y1_din_from_l1;
        end else begin
            // L2 owns Y1 (reading via x_*); force WE=0
            y1_cs = x1_cs_from_l2; y1_we = 1'b0; y1_a = x1_addr_from_l2; y1_d = '0;
        end
    end

    // ---- Y2 mux: before L2 finishes → L2 writes; after → L3 reads
    always @(*) begin
        if (!l2_done_seen) begin
            y2_cs = y2_cs_from_l2; y2_we = y2_we_from_l2; y2_a = y2_addr_from_l2; y2_d = y2_din_from_l2;
        end else begin
            y2_cs = x2_cs_from_l3; y2_we = 1'b0; y2_a = x2_addr_from_l3; y2_d = '0;
        end
    end

    // ---- Y3 mux: before L3 finishes → L3 writes; after → ARG reads (WE=0)
    always @(*) begin
        if (!l3_done_seen) begin
            y3_cs = y3_cs_from_l3; y3_we = y3_we_from_l3; y3_a = y3_addr_from_l3; y3_d = y3_din_from_l3;
        end else begin
            y3_cs = y3_cs_from_arg; y3_we = 1'b0;          y3_a = y3_addr_from_arg; y3_d = '0;
        end
    end

endmodule
