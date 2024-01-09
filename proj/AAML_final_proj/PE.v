module PE(
    clk,
    rst,
    offset,
    inp_left,
    inp_up,
    out_right,
    out_down,
    answer
);

input clk, rst;
input [31:0] offset;
input [7:0] inp_left, inp_up;
output reg [7:0] out_right, out_down;
output reg signed [31:0] answer;
wire signed [15:0] mult;
reg signed [8:0] inp_left_d;
reg signed [7:0] inp_up_d;
reg signed [15:0] mult_0, mult_1;

always @(negedge rst or posedge clk) begin
    if(!rst) begin
        answer <= 0;
        out_right <= 0;
        out_down <= 0;
    end
    else begin
        inp_left_d <= $signed(inp_left) + $signed(offset[8:0]);
        inp_up_d <= $signed(inp_up);
        mult_0 <= inp_left_d * inp_up_d;
        mult_1 <= mult_0;
        answer <= answer + mult_1;
        out_right <= inp_left;
        out_down <= inp_up;
    end
end

// assign mult = (enable) ? $signed(inp_up) * ($signed(inp_left) + $signed(offset[8:0])) : 0;

endmodule
