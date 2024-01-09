// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "global_buffer.v"
`include "PE.v"
`include "systolic_array.v"
`include "TPU.v"

module Cfu (
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output reg          rsp_valid,
  input               rsp_ready,
  output reg [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);

  reg          in_valid;
  reg          reset_tpu;
  reg          reset_acc;
  reg [8:0]    K;
  reg [7:0]    M;
  reg [7:0]    N;
  wire         busy;


  wire              A_wr_en;
  wire              A_wr_en_tpu;
  reg               A_wr_en_cfu;
  wire     [7:0]    A_index;
  wire     [7:0]    A_index_tpu;
  reg      [7:0]    A_index_cfu;
  wire     [31:0]   A_data_in;
  wire     [31:0]   A_data_in_tpu;
  reg      [31:0]   A_data_in_cfu;
  wire     [31:0]   A_data_out;

  wire              B_wr_en;
  wire              B_wr_en_tpu;
  reg               B_wr_en_cfu;
  wire     [7:0]    B_index;
  wire     [7:0]    B_index_tpu;
  reg      [7:0]    B_index_cfu;
  wire     [31:0]   B_data_in;
  wire     [31:0]   B_data_in_tpu;
  reg      [31:0]   B_data_in_cfu;
  wire     [31:0]   B_data_out;

  wire              C_wr_en;
  wire     [1:0]    C_index;
  wire     [1:0]    C_index_tpu;
  reg      [1:0]    C_index_cfu;
  wire     [127:0]  C_data_in;
  wire     [127:0]  C_data_out;

  reg [6:0] cmd_code;
  reg [1:0] C_idx;
  reg [1:0] tpu_work_state;
  reg [31:0] input_offset;
  reg [1:0] C_rd_state;
  reg delay_reg;
  reg [1:0] tpu_work_state_next;

  assign A_wr_en = (tpu_work_state != 0) ? A_wr_en_tpu : A_wr_en_cfu;
  assign A_index = (tpu_work_state != 0) ? A_index_tpu : A_index_cfu;
  assign A_data_in = (tpu_work_state != 0) ? A_data_in_tpu : A_data_in_cfu;
  assign B_wr_en = (tpu_work_state != 0) ? B_wr_en_tpu : B_wr_en_cfu;
  assign B_index = (tpu_work_state != 0) ? B_index_tpu : B_index_cfu;
  assign B_data_in = (tpu_work_state != 0) ? B_data_in_tpu : B_data_in_cfu;
  assign C_index = (tpu_work_state != 0) ? C_index_tpu : C_index_cfu;

  global_buffer#(.ADDR_BITS(8), .DATA_BITS(32))
  gbuff_A (
    .clk(clk),
    .rst_n(~reset),
    .wr_en(A_wr_en),
    .index(A_index),
    .data_in(A_data_in),
    .data_out(A_data_out)
  );

  global_buffer #(.ADDR_BITS(8), .DATA_BITS(32))
  gbuff_B(
      .clk(clk),
      .rst_n(~reset),
      .wr_en(B_wr_en),
      .index(B_index),
      .data_in(B_data_in),
      .data_out(B_data_out)
  );


  global_buffer #(.ADDR_BITS(2), .DATA_BITS(128))
  gbuff_C(
      .clk(clk),
      .rst_n(~reset),
      .wr_en(C_wr_en),
      .index(C_index),
      .data_in(C_data_in),
      .data_out(C_data_out)
  );

  TPU My_TPU(
      .clk            (clk),     
      .rst_n          (reset_tpu),     
      .rst_acc        (reset_acc),
      .offset         (input_offset),
      .in_valid       (in_valid),         
      .K              (K), 
      .M              (M), 
      .N              (N), 
      .busy           (busy),     
      .A_wr_en        (A_wr_en_tpu),         
      .A_index        (A_index_tpu),         
      .A_data_in      (A_data_in_tpu),         
      .A_data_out     (A_data_out),         
      .B_wr_en        (B_wr_en_tpu),         
      .B_index        (B_index_tpu),         
      .B_data_in      (B_data_in_tpu),         
      .B_data_out     (B_data_out),         
      .C_wr_en        (C_wr_en),         
      .C_index        (C_index_tpu),         
      .C_data_in      (C_data_in),         
      .C_data_out     (C_data_out)         
  );


  // Only not ready for a command when we have a response.
  assign cmd_ready = ~rsp_valid;

  always @(posedge clk) begin
    if (tpu_work_state == 1) begin
      in_valid <= 1;
    end
    else if (tpu_work_state == 2) begin
      in_valid <= 0;
    end
  end

  always @(posedge clk) begin
    if (reset)
      tpu_work_state = 0;
    else
      tpu_work_state = tpu_work_state_next;
  end

  always @(posedge clk) begin
    if (reset) begin
      rsp_valid <= 0;
      delay_reg <= 0;
      A_wr_en_cfu <= 0;
      A_data_in_cfu <= 0;
      A_index_cfu <= 0;
      B_wr_en_cfu <= 0;
      B_data_in_cfu <= 0;
      B_index_cfu <= 0;
      C_rd_state <= 0;
      rsp_payload_outputs_0 <= 32'b0;
      reset_tpu <= 1;
      reset_acc <= 1;
    end else if (delay_reg) begin
      rsp_valid <= 1;
      delay_reg <= 0;
      A_wr_en_cfu <= 0;
      A_data_in_cfu <= 0;
      A_index_cfu <= 0;
      B_wr_en_cfu <= 0;
      B_data_in_cfu <= 0;
      B_index_cfu <= 0;
      rsp_payload_outputs_0 <= cmd_code;
      reset_tpu <= 1;
      reset_acc <= 1;
    end else if (rsp_valid) begin
      // Waiting to hand off response to CPU.
      rsp_valid <= ~rsp_ready;
    end else if (cmd_valid) begin
      cmd_code <= cmd_payload_function_id[9:3];
      case(cmd_payload_function_id[9:3])
        1: begin  // write A
          A_wr_en_cfu <= 1;
          A_index_cfu <= cmd_payload_inputs_0[7:0];
          A_data_in_cfu <= cmd_payload_inputs_1;
          delay_reg <= 1;
        end
        2: begin  // write B
          B_wr_en_cfu <= 1;
          B_index_cfu <= cmd_payload_inputs_0[7:0];
          B_data_in_cfu <= cmd_payload_inputs_1;
          delay_reg <= 1;
        end
        3: begin  // read C
          C_index_cfu <= cmd_payload_inputs_0[1:0];
          C_idx <= cmd_payload_inputs_1[1:0];
          C_rd_state <= 1;
        end
        4: begin  // passing K
          K <= cmd_payload_inputs_0[8:0];
          rsp_payload_outputs_0 <= 4;
          rsp_valid <= 1;
        end
        5: begin  // perform calculation
          tpu_work_state_next <= 1;
        end
        6: begin  // reset TPU except the accumulator
          reset_tpu <= 0;
          delay_reg <= 1;
        end
        7: begin  // reset TPU and its accumulator
          reset_tpu <= 0;
          reset_acc <= 0;
          delay_reg <= 1;
        end
        8: begin  // passing input offset
          input_offset <= cmd_payload_inputs_0;
          rsp_payload_outputs_0 <= 8;
          rsp_valid <= 1;
        end
        default: begin
          rsp_payload_outputs_0 <= -1;
          rsp_valid <= 1;
        end
      endcase
    end
    else if (tpu_work_state) begin
      case(tpu_work_state)
        0: tpu_work_state_next = 0;
        1: tpu_work_state_next = 2;
        2: tpu_work_state_next = 3;
        3: begin
          if (busy)
            tpu_work_state_next = 3;
          else begin
            tpu_work_state_next <= 0;
            rsp_payload_outputs_0 <= 5;
            rsp_valid <= 1;
          end
        end
      endcase
    end
    else if (C_rd_state) begin
      case(C_rd_state)
        0: C_rd_state <= 0;
        1: C_rd_state <= 2;
        2: begin
          C_rd_state <= 0;
          rsp_payload_outputs_0 <= C_data_out[C_idx[1:0]*32+:32];
          rsp_valid <= 1;
        end
        3: C_rd_state <= 0;
      endcase
    end
  end
endmodule