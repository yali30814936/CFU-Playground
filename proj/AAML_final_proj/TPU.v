module TPU(
    clk,
    rst_n,
    rst_acc,
    offset,

    in_valid,
    K,
    M,
    N,
    busy,

    A_wr_en,
    A_index,
    A_data_in,
    A_data_out,

    B_wr_en,
    B_index,
    B_data_in,
    B_data_out,

    C_wr_en,
    C_index,
    C_data_in,
    C_data_out
);


input clk;
input rst_n;
input rst_acc;
input [31:0]     offset;
input            in_valid;
input [8:0]      K;
input [7:0]      M;
input [7:0]      N;
output           busy;

output           A_wr_en;
output [7:0]     A_index;
output [31:0]    A_data_in;
input  [31:0]    A_data_out;

output           B_wr_en;
output [7:0]     B_index;
output [31:0]    B_data_in;
input  [31:0]    B_data_out;

output           C_wr_en;
output [1:0]     C_index;
output [127:0]   C_data_in;
input  [127:0]   C_data_out;



//* Implement your design here

parameter IDLE = 2'd0;
parameter BUSY = 2'd1;
parameter OUTP = 2'd2;
parameter DONE = 2'd3;

parameter ARRAY_SIZE = 4;
parameter ARRAY_WIDTH= ARRAY_SIZE * 8;
reg [ARRAY_WIDTH-1:0] A[0:ARRAY_SIZE*2-1], B[0:ARRAY_SIZE*2-1];
wire [ARRAY_WIDTH*4 * ARRAY_SIZE-1:0] C_w;
reg [ARRAY_WIDTH*4-1:0] C_data_in_r;
reg [31:0] inp_left, inp_up;
reg [2:0] mM, mN;
reg [8:0] mK;
reg [8:0] counter;
reg [2:0] out_counter;
reg [8:0] load_index;
reg first_busy;
reg first_out;
reg [2:0] out_index;
reg [1:0] state, n_state;
integer i, j;

assign A_index = load_index;
assign B_index = load_index;

systolic_array SARR(
    .clk(clk),
    .rst(rst_acc),
    .offset(offset),
    .in_left(inp_left),
    .in_up(inp_up),
    .ans(C_w)
);



assign C_index = out_index;
assign C_wr_en = (state == OUTP && out_index < 4);
assign C_data_in = C_data_in_r;

assign busy = (in_valid || state == BUSY || state == OUTP);

always @(*) begin
    case (state)
        IDLE: n_state = (in_valid) ? BUSY : IDLE;
        BUSY: n_state = (counter < mK + 8) ? BUSY : OUTP;
        OUTP: n_state = (out_counter < 5) ? OUTP : DONE;
        DONE: n_state = DONE;
        default: n_state = IDLE;
    endcase
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        state = IDLE;
    else
        state = n_state;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        {mM, mK, mN} = 0;
    end
    else if (in_valid) begin
        mM <= M;
        mK <= K;
        mN <= N;
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        load_index <= 0;
    else begin
        if (n_state == BUSY)
            load_index <= load_index + 1;
    end
end

always @(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    counter <= 0;
    first_busy <= 1;
  end
  else begin
    if(state == BUSY || state == OUTP)
        if (first_busy)
            first_busy <= 0;
        else
            counter <= counter + 1;
    else
      counter <= 0;
  end
end

always @(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    out_counter <= 0;
    first_out <= 1;
    out_index <= 0;
  end
  else begin
    if(state == OUTP) begin
        out_counter <= out_counter + 1;
        if (first_out)
            first_out <= 0;
        else
            out_index <= out_index + 1;
    end
    else begin
      out_counter <= 0;
      out_index <= 0;
    end
  end
end

///////////////////////////////////////////////////////////
//                                                       //
//                    Data_Loader                        //
//                                                       //
///////////////////////////////////////////////////////////

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin 
        for (i=0; i<ARRAY_SIZE*2; i = i+1) begin
            A[i] <= 0;
            B[i] <= 0;
        end
    end
    else if (load_index > 0 && load_index <= mK) begin
        A[(load_index-1) & 3'd7] <= A_data_out;
        B[(load_index-1) & 3'd7] <= B_data_out;
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        inp_left <= 0;
        inp_up <= 0;
    end
    else if ((state == BUSY && counter > 0) || state == OUTP)begin
        for (j=0; j<ARRAY_SIZE; j = j+1) begin
            if((counter-1 + j > 2) && (counter-1 + j < mK+3)) begin
                inp_left[(j<<3)+:8] <= A[(counter-1-3+j) & 3'd7][j<<3+:8];
                inp_up[(j<<3)+:8] <= B[(counter-1-3+j) & 3'd7][j<<3+:8];
            end
            else begin
                inp_left[(j<<3)+:8] <= 0;
                inp_up[(j<<3)+:8] <= 0;
            end
        end
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) C_data_in_r <= 0;
    else begin
        if (n_state == OUTP)
            C_data_in_r <= C_w[out_counter*ARRAY_WIDTH*4+:ARRAY_WIDTH*4];
        else
            C_data_in_r <= 0;
    end
end

endmodule