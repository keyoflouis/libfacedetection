#include "cuda_fonc.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define thread_per_block 256

CDataBlobKernel *convolution_1x1pointwiseKernel(int input_rows,
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
                                                float *output_data){
	
	CDataBlobKernel inputData(input_rows, input_cols, input_channels, input_channelStep, input_data);
	CDataBlobKernel outputData(output_rows, output_cols, output_channels, output_channelStep, output_data);
	FiltersKernel filters(  channels, num_filters, is_depthwise, is_pointwise, with_relu,
                      		weights_rows, weights_cols, weights_channels, weights_channelStep, weight_data,
                      		biases_rows, biases_cols, biases_channels, biases_channelStep, biases_data);



	
	CDataBlobKernel* dev_inputData;
	CDataBlobKernel* dev_outputData;
	FiltersKernel* dev_filters;


	cudaMallocManaged((void**)&dev_inputData,sizeof(CDataBlobKernel));
	cudaMallocManaged((void**)&dev_outputData,sizeof(CDataBlobKernel));
	cudaMallocManaged((void**)&dev_filters,sizeof(FiltersKernel));
	
	
	float* dev_input_data;
	float* dev_output_data;

	// allocate and deep copy for inpute
 	size_t size_bytes_devInputData = size_t(inputData.rows) * inputData.cols * inputData.channelStep;
	cudaMallocManaged((void**)&dev_input_data, size_bytes_devInputData);
	cudaMemcpy(dev_input_data ,inputData.data, size_bytes_devInputData,cudaMemcpyHostToDevice);
	cudaMemcpy( &dev_inputData->data , &dev_input_data , sizeof(float*) , cudaMemcpyHostToDevice );

	// allocate and deep copy for outpute
	size_t size_bytes_devOutputData = size_t(outputData.rows) * outputData.cols * outputData.channelStep;
	cudaMallocManaged((void**)&dev_output_data , size_bytes_devOutputData );
	cudaMemcpy(dev_output_data, outputData.data, size_bytes_devOutputData, cudaMemcpyHostToDevice);
	cudaMemcpy( &dev_outputData->data , &dev_output_data , sizeof(float*) , cudaMemcpyHostToDevice );


	
	// allocate and deep copy for filters
	float* dev_filters_weightsData;
	float* dev_filters_biasesData;
	
	size_t size_bytes_devfilters_weightsData = size_t(filters.weights.rows) * filters.weights.cols * filters.weights.channelStep;
	size_t size_bytes_devfilters_biasesData = size_t(filters.biases.rows) * filters.biases.cols * filters.biases.channelStep;

	cudaMallocManaged((void**)&dev_filters_weightsData,size_bytes_devfilters_weightsData);
	cudaMallocManaged((void**)&dev_filters_biasesData,size_bytes_devfilters_biasesData);
	
	cudaMemcpy(dev_filters_weightsData ,  filters.weights.data , size_bytes_devfilters_weightsData ,cudaMemcpyHostToDevice);
	cudaMemcpy(&dev_filters->weights.data,&dev_filters_weightsData, size_bytes_devfilters_weightsData,cudaMemcpyHostToDevice);
	
	cudaMemcpy(dev_filters_biasesData ,  filters.biases.data , size_bytes_devfilters_biasesData ,cudaMemcpyHostToDevice);
	cudaMemcpy(&dev_filters->biases.data,&dev_filters_biasesData, size_bytes_devfilters_biasesData,cudaMemcpyHostToDevice);

	// invoke	


	// store the results
	cudaMemcpy(&outputData , dev_outputData,sizeof(CDataBlobKernel),cudaMemcpyDeviceToHost);
	cudaMemcpy(&outputData.data , &dev_outputData->data,size_bytes_devOutputData ,cudaMemcpyDeviceToHost);

	// free
	
	cudaFree(dev_input_data);
	cudaFree(dev_inputData->data);
	cudaFree(dev_inputData);

	cudaFree(dev_output_data);
	cudaFree(dev_outputData->data);
	cudaFree(dev_outputData);

	cudaFree(dev_filters_weightsData);
	cudaFree(dev_filters_biasesData);
	cudaFree(dev_filters->biases.data);
	cudaFree(dev_filters->weights.data);
	cudaFree(dev_filters);

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