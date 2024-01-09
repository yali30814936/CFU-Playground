module systolic_array(
    input clk,
    input rst,
    input [31:0] offset,
    input [31:0] in_left,
    input [31:0] in_up,
    output [511:0] ans
);

    wire [31:0] hwire[0:4], vwire[0:4];
    assign hwire[0] = in_left;
    assign vwire[0] = in_up;

    genvar i, j;
    generate
        for (i=0; i<4; i = i+1) begin: SARR_rows
            for (j=0; j<4; j = j+1) begin: SARR_cols
                PE pe(
                    .clk(clk),
                    .rst(rst),
                    .offset(offset),
                    .inp_left(hwire[j][(3-i)*8+:8]),
                    .inp_up(vwire[i][(3-j)*8+:8]),
                    .out_right(hwire[j+1][(3-i)*8+:8]),
                    .out_down(vwire[i+1][(3-j)*8+:8]),
                    .answer(ans[i*128+(3-j)*32+:32])
                );
            end
        end
    endgenerate
endmodule