#include "cuda_fonc.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define thread_per_block 256

CDataBlobKernel* convolution_1x1pointwiseKernel(int input_rows,
                                    int input_cols,
                                    int input_channels,
                                    int input_channelStep,
                                    float *input_data,

                                    int t_channels,
                                    int num_filters,
                                    bool is_depthwise,
                                    bool is_pointwise,
                                    bool with_relu,

                                    int weights_rows,
                                    int weights_cols,
                                    int weights_channels,
                                    int weights_channelStep,
                                    float *weight_data,

                                    int biases_rows,
                                    int biases_cols,
                                    int biases_channels,
                                    int biases_channelStep,
                                    float *biases_data,

                                    int output_rows,
                                    int output_cols,
                                    int output_channels,
                                    int output_channelStep,
                                    float *output_data)
{
    CDataBlobKernel inputData(input_rows, input_cols, input_channels, input_channelStep, input_data);
    CDataBlobKernel outputData(output_rows, output_cols, output_channels, output_channelStep, output_data);
    FiltersKernel filters(t_channels, num_filters, is_depthwise, is_pointwise, with_relu,

                          weights_rows, weights_cols, weights_channels, weights_channelStep, weight_data,

                          biases_rows, biases_cols, biases_channels, biases_channelStep, biases_data);

    for (int row = 0; row < outputData.rows; row++)
    {
        for (int col = 0; col < outputData.cols; col++)
        {
            float *pOut = outputData.ptr(row, col);
            const float *pIn = inputData.ptr(row, col);
            for (int ch = 0; ch < outputData.channels; ch++)
            {
                const float *pF = filters.weights.ptr(0, ch);
                float sum = 0.f;
                for (int i = 0; i < inputData.channels; i++)
                {
                    sum += (pIn[i] * pF[i]);
                }
                pOut[ch] = sum;
                pOut[ch] += filters.biases.data[ch];
            }
        }
    }
    return &outputData;
};