#include"cuda_fonc.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>

#define thread_per_block 256

bool convolution_1x1pointwiseKernel(int input_rows, 
									int input_cols, 
									int input_channels, 
									int input_channelStep, 
									float *input_data,


									int channels,
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
									float* output_data) 
{
	return true;
};