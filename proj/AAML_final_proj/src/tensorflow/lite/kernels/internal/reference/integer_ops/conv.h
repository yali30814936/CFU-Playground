/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>
#include "cfu.h"
#include <cstdio>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

namespace tflite {
namespace reference_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  // const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int32_t input_offset = 128;  // r = s(q - Z)
  const int32_t neg_input_offset = -128;
  const uint64_t neg_input_offset_vec = 0x8080808080808080;
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  // const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  // const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  int8_t input_mat[0xffffff];
  int8_t filter_mat[0xffffff];
  int32_t output_mat[0xffffff];
for (int batch = 0; batch < batches; ++batch) {
    int8_t *input_ptr = input_mat;
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
            const int in_x = in_x_origin + dilation_width_factor * filter_x;

            // Zero padding by omitting the areas outside the image.
            const bool is_point_inside_image =
                (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                (in_y < input_height);

            if (filter_input_depth & 7)
              for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
                if (!is_point_inside_image)
                  *(input_ptr++) = neg_input_offset;
                else
                  *(input_ptr++) = input_data[Offset(input_shape, batch, in_y, in_x, in_channel)];
              }
            else
              for (int in_channel = 0; in_channel < filter_input_depth; in_channel+=8) {
                if (!is_point_inside_image)
                  *((uint64_t*)input_ptr) = neg_input_offset_vec;
                else
                  *((uint64_t*)input_ptr) = *((uint64_t*)(input_data + Offset(input_shape, batch, in_y, in_x, in_channel)));
                input_ptr += 8;
              }
          }
        }
      }
    }

    int8_t *filter_ptr = filter_mat;
    for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
      for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
          if (filter_input_depth & 7)
            for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel)
              *(filter_ptr++) = filter_data[Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)];
          else
            for (int in_channel = 0; in_channel < filter_input_depth; in_channel+=8) {
              *((uint64_t*)filter_ptr) = *((uint64_t*)(filter_data + Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)));
              filter_ptr += 8;
            }
        }
      }
    }

    

    // int32_t *output_ptr = output_mat;
    // for (int in_y=0; in_y < output_height*output_width; in_y++) {
    //   for (int kn_y=0; kn_y < output_depth; ++kn_y) {
    //     int32_t acc = 0;
    //     for (int i=0; i<filter_height*filter_width*filter_input_depth; ++i) {
    //       int32_t input_val = input_mat[in_y*filter_height*filter_width*filter_input_depth + i];
    //       int32_t filter_val = filter_mat[kn_y*filter_height*filter_width*filter_input_depth + i];
    //       acc += filter_val * (input_val + input_offset);
    //     }

    //     *(output_ptr++) = acc;
    //   }
    // }

    int bigM = output_height*output_width;
    int bigK = filter_height*filter_width*filter_input_depth;
    int bigN = output_depth;
    static int layer = 0;

    // if (layer == 1) {
    //   for (int i=0; i<16; ++i) {
    //     for (int j=0; j<std::min(32, bigK); ++j) {
    //       printf("%02x ", filter_mat[i*bigK + j] & 0xff);
    //     }
    //     printf("\n");
    //   }
    //   printf("\n");
    // }

    // printf("%d: %dx%d(%d) * %dx%d(%d)\n", layer, bigM, bigK, input_ptr-input_mat, bigK, bigN, filter_ptr-filter_mat);
    int ret = cfu_op0(8, input_offset, 0);  // passing offset
    if (ret != 8)  {
      printf("Layer %d(%dx%dx%d): passing offset return status %d\n", layer, bigM, bigK, bigN, ret);
      return;
    }
    
    for (int in_y=0; in_y < bigM; in_y+=4) {
      for (int kn_y=0; kn_y < bigN; kn_y+=4) {
        ret = cfu_op0(7, 0, 0); // reset TPU and its accumulator
        if (ret != 7) {
          printf("Layer %d(%dx%dx%d): in_y=%d, kn_y=%d, reset tpu & acc, return %d\n", layer, bigM, bigK, bigN, in_y, kn_y, ret);
          return;
        }
        for (int i=0; i<bigK; i+=256) {
          for (int j=0; j<std::min(bigK-i, 256); ++j) {
            uint32_t tmp = 0;
            for (int k=0; k<4; ++k) {
              if (in_y+k < bigM)
                tmp |= uint32_t(uint8_t(input_mat[(in_y+k)*bigK + i + j])) << (3-k)*8;
            }

            ret = cfu_op0(1, j, tmp); // write A
            if (ret != 1) {
              printf("Layer %d(%dx%dx%d): in_y=%d, kn_y=%d, i=%d, j=%d(<%d), writing A return %d\n", layer, bigM, bigK, bigN, in_y, kn_y, i, j, std::min(bigK-i, 256), ret);
              return;
            }

            tmp = 0;
            for (int k=0; k<4; ++k) {
              if (kn_y+k < bigN)
                tmp |= uint32_t(uint8_t(filter_mat[(kn_y+k)*bigK + i + j])) << (3-k)*8;
            }
              
            ret = cfu_op0(2, j, tmp); // write B
            if (ret != 2) {
              printf("Layer %d(%dx%dx%d): in_y=%d, kn_y=%d, i=%d, j=%d(<%d), writing B return %d\n", layer, bigM, bigK, bigN, in_y, kn_y, i, j, std::min(bigK-i, 256), ret);
              return;
            }
          }
          ret = cfu_op0(4, std::min(bigK-i, 256), 0);  // passing K
          if (ret != 4) {
            printf("Layer %d(%dx%dx%d): in_y=%d, kn_y=%d, i=%d, passing K=%d return %d\n", layer, bigM, bigK, bigN, in_y, kn_y, i, std::min(bigK-i, 256), ret);
            return;
          }
          ret = cfu_op0(6, 0, 0); //reset TPU except the accumulator
          if (ret != 6) {
            printf("Layer %d(%dx%dx%d): in_y=%d, kn_y=%d, i=%d, reset tpu except acc, return %d\n", layer, bigM, bigK, bigN, in_y, kn_y, i, ret);
            return;
          }
          ret = cfu_op0(5, 0, 0);  // start calculation
          if (ret != 5) {
            printf("Layer %d(%dx%dx%d): in_y=%d, kn_y=%d, i=%d, open fire return status %d\n", layer, bigM, bigK, bigN, in_y, kn_y, i, ret);
            return;
          }
        }

        for (int i=0; i<4; ++i) {
          for (int j=0; j<4; ++j) {
            if ((in_y + i) < bigM && (kn_y + j) < bigN) {
              int val = cfu_op0(3, i, 3-j); // read C;
              output_mat[(in_y+i)*bigN + (kn_y+j)] = val;
            }
          }
        }
      }
    }

    int32_t *output_ptr = output_mat;
    // output_ptr = output_mat;
    for (int out_y=0; out_y < output_height; ++out_y) {
      for (int out_x=0; out_x < output_width; ++out_x) {
        for (int out_chan=0; out_chan < output_depth; ++out_chan) {
          int32_t acc = *(output_ptr++);
          // if (layer == 0 && out_y==0 && out_x<4) {
          //   if (out_chan<4)
          //     printf("%08lx ", acc);
          //   else if (out_chan==4)
          //     printf("\n");
          // }
          if (bias_data) {
            acc += bias_data[out_chan];
          }
          acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_chan], output_shift[out_chan]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_chan)] =
              static_cast<int8_t>(acc);
        }
      }
    }
    ++layer;
  }
}

inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
