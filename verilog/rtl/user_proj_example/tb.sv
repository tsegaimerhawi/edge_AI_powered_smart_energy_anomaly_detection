`timescale 1ns/1ps

module tb_nn_7_32_16_4;
    // ---------------- Clock/Reset ----------------
    reg clk = 0;
    always #5 clk = ~clk;   // 100 MHz

    reg rst_n = 0;

    // --------------- DUT I/O ---------------------
    reg         start;
    reg         x0_wr_en;
    reg  [2:0]  x0_wr_addr;
    reg  signed [15:0] x0_wr_data;

    wire        done;
    wire [1:0]  class_out;
    wire signed [15:0] score_max;

    // --------------- DUT ------------------------
    nn_top_csw_7_32_16_4 #(
        .DW(16),
        .ACC_W(40)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),

        .x0_wr_en(x0_wr_en),
        .x0_wr_addr(x0_wr_addr),
        .x0_wr_data(x0_wr_data),

        .done(done),
        .class_out(class_out),
        .score_max(score_max)
    );

    // --------------- TB memories for loading ---------------
    // Layer-1: B(32 x 16), W(224 x 8)
    reg signed [15:0] l1b [0:31];
    reg        [7:0]  l1w [0:223];

    // Layer-2: B(16 x 16), W(512 x 8)
    reg signed [15:0] l2b [0:15];
    reg        [7:0]  l2w [0:511];

    // Layer-3: B(4 x 16), W(64 x 8)
    reg signed [15:0] l3b [0:3];
    reg        [7:0]  l3w [0:63];

    // Input vector X0 (7 x 16)
    reg signed [15:0] x0v [0:6];

    integer i;

    // --------------- Tasks: program DUT RAMs via hierarchy ---------------

    task load_layer1_params;
        integer k;
        begin
            // Biases B1[0..31]
            for (k = 0; k < 32; k = k+1) begin
                dut.B1.mem[k] = l1b[k];
            end

            // Weights W1[0..223], sign-extend 8->16, contiguous addresses (n*7 + i)
            for (k = 0; k < 224; k = k+1) begin
                dut.W1.mem[k] = {{8{l1w[k][7]}}, l1w[k]};
            end

            // Clear unused rows if any (224..255)
            for (k = 224; k < 256; k = k+1) begin
                dut.W1.mem[k] = 16'sd0;
            end
        end
    endtask

    task load_layer2_params;
        integer k;
        begin
            // Biases B2[0..15]
            for (k = 0; k < 16; k = k+1) begin
                dut.B2.mem[k] = l2b[k];
            end

            // Weights W2[0..511], sign-extend 8->16
            for (k = 0; k < 512; k = k+1) begin
                dut.W2.mem[k] = {{8{l2w[k][7]}}, l2w[k]};
            end
        end
    endtask

    task load_layer3_params;
        integer k;
        begin
            // Biases B3[0..3]
            for (k = 0; k < 4; k = k+1) begin
                dut.B3.mem[k] = l3b[k];
            end

            // Weights W3[0..63], sign-extend 8->16
            for (k = 0; k < 64; k = k+1) begin
                dut.W3.mem[k] = {{8{l3w[k][7]}}, l3w[k]};
            end
        end
    endtask

    task load_input_x0;
        integer k;
        begin
            for (k = 0; k < 7; k = k+1) begin
                // Use the DUT’s host write interface for X0 (keeps behavior realistic)
                @(negedge clk);
                x0_wr_en   = 1'b1;
                x0_wr_addr = k[2:0];
                x0_wr_data = x0v[k];
                @(negedge clk);
                x0_wr_en   = 1'b0;
            end
        end
    endtask

    // --------------- Main stimulus ---------------
    initial begin
        // Defaults
        start     = 1'b0;
        x0_wr_en  = 1'b0;
        x0_wr_addr= '0;
        x0_wr_data= '0;

        // Hold reset a bit
        rst_n = 1'b0;
        repeat (5) @(negedge clk);
        rst_n = 1'b1;

        // Read hex files
        // (Files’ exact contents are provided below this TB—copy/paste to .hex files)
        $readmemh("C:\Users\Adam\Desktop\Z_anolmaly_detection\layer1b.hex", l1b);
        $readmemh("C:\Users\Adam\Desktop\Z_anolmaly_detection\layer1w.hex", l1w);
        $readmemh("C:\Users\Adam\Desktop\Z_anolmaly_detection\layer2b.hex", l2b);
        $readmemh("C:\Users\Adam\Desktop\Z_anolmaly_detection\layer2w.hex", l2w);
        $readmemh("C:\Users\Adam\Desktop\Z_anolmaly_detection\layer3b.hex", l3b);
        $readmemh("C:\Users\Adam\Desktop\Z_anolmaly_detection\layer3w.hex", l3w);
        $readmemh("C:\Users\Adam\Desktop\Z_anolmaly_detection\x0.hex",     x0v);


        // Program DUT internal SRAMs
        load_layer1_params();
        load_layer2_params();
        load_layer3_params();

        // Load input vector X0 (7 elements)
        load_input_x0();

        // Start inference
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        // Wait for done
        wait (done === 1'b1);
        @(negedge clk);

        $display("=====================================================");
        $display("Inference done. class_out = %0d, score_max = %0d (0x%h)",
                  class_out, score_max, score_max);
        $display("Y3 contents:");
        for (i = 0; i < 4; i = i+1)
            $display("  Y3[%0d] = %0d (0x%h)", i, $signed(dut.Y3.mem[i]), dut.Y3.mem[i]);
        $display("=====================================================");

        // A couple extra cycles then finish
        repeat (5) @(negedge clk);
        $finish;
    end

endmodule
