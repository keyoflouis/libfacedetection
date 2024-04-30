#include "cuda_fonc.cuh"
#ifndef CUDA_HEADER
#define CUDA_HEADER	

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif 

#include <stdio.h>

__global__ void kernel(CDataBlobKernel* inputData, CDataBlobKernel* outputData, FiltersKernel* filters) {
	
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

		 //parallel calculate every group of the input
		 if(row < outputData->rows && col < outputData->cols)
			{
				float* pOut = outputData->ptr(row, col);
				const float* pIn = inputData->ptr(row, col);

				// calculate the element in a group
				for (int ch = 0; ch < outputData->channels; ch++)
				{
					const float* pF = filters->weights.ptr(0, ch);
					float sum = 0.f;
					for (int i = 0; i < inputData->channels; i++)
					{
						sum += (pIn[i] * pF[i]);
					}
					pOut[ch] = (sum + filters->biases.data[ch]);
				}		
			}
}

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


	
	
	oopBlobToKernel input(inputData);
	oopBlobToKernel output(outputData);


	FiltersKernel* dev_filters;
	cudaMalloc((void**)&dev_filters,sizeof(FiltersKernel));

	// allocate and deep copy for filters
	float* dev_filters_weightsData;
	float* dev_filters_biasesData;
	cudaMemcpy(dev_filters, &filters,sizeof(FiltersKernel),cudaMemcpyHostToDevice);


	size_t size_bytes_devfilters_weightsData = size_t(filters.weights.rows) * filters.weights.cols * filters.weights.channelStep;
	size_t size_bytes_devfilters_biasesData = size_t(filters.biases.rows) * filters.biases.cols * filters.biases.channelStep;

	cudaMalloc((void**)&dev_filters_weightsData,size_bytes_devfilters_weightsData);
	cudaMalloc((void**)&dev_filters_biasesData,size_bytes_devfilters_biasesData);
	
	cudaMemcpy(dev_filters_weightsData ,  filters.weights.data , size_bytes_devfilters_weightsData ,cudaMemcpyHostToDevice);
	cudaMemcpy(&dev_filters->weights.data,&dev_filters_weightsData, sizeof(float*),cudaMemcpyHostToDevice);
	
	cudaMemcpy(dev_filters_biasesData ,  filters.biases.data , size_bytes_devfilters_biasesData ,cudaMemcpyHostToDevice);
	cudaMemcpy(&dev_filters->biases.data,&dev_filters_biasesData, sizeof(float*),cudaMemcpyHostToDevice);

	// invoke

	int blockSize = 16;

	int gridRows = (outputData.rows + blockSize - 1) / blockSize;
	int gridCols = (outputData.cols + blockSize - 1) / blockSize;

	dim3 dimBlock(blockSize, blockSize);
	dim3 dimGrid(gridCols, gridRows); 

	kernel << <dimGrid, dimBlock >> > (input.devCDataBlob, output.devCDataBlob, dev_filters);

	cudaDeviceSynchronize();

	// store the results
	void* temp = outputData.data;
	cudaMemcpy(&outputData , output.devCDataBlob,sizeof(CDataBlobKernel),cudaMemcpyDeviceToHost);
	outputData.data = (float*)temp;
	cudaMemcpy(outputData.data, output.devdata, output.size_inbytes_Data, cudaMemcpyDeviceToHost);


	// free

	cudaFree(dev_filters_weightsData);
	cudaFree(dev_filters_biasesData);
	cudaFree(dev_filters);

    return &outputData;
};