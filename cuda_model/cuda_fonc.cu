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
                                                float *output_data)
{
    CDataBlobKernel inputData(input_rows, input_cols, input_channels, input_channelStep, input_data);
    CDataBlobKernel outputData(output_rows, output_cols, output_channels, output_channelStep, output_data);

    FiltersKernel filters(channels, num_filters, is_depthwise, is_pointwise, with_relu,
                          weights_rows, weights_cols, weights_channels, weights_channelStep, weight_data,
                          biases_rows, biases_cols, biases_channels, biases_channelStep, biases_data);
    
    CDataBlobKernel *host_inputData = &inputData;
    CDataBlobKernel *host_outputData = &outputData;
    FiltersKernel *host_filters = &filters;

    CDataBlobKernel *dev_inputData =nullptr;
    CDataBlobKernel *dev_outputData =nullptr;
    FiltersKernel *dev_filters =nullptr;

    cudaMallocManaged((void **)&dev_inputData, sizeof(CDataBlobKernel));
    cudaMallocManaged((void **)&dev_outputData, sizeof(CDataBlobKernel));
    cudaMallocManaged((void **)&dev_filters, sizeof(FiltersKernel));

    // define the device pointer
    cudaMemcpy((void **)&dev_inputData, (void **)&host_inputData, sizeof(CDataBlobKernel), cudaMemcpyHostToDevice);
    cudaMemcpy((void **)&dev_outputData, (void **)&host_outputData, sizeof(CDataBlobKernel), cudaMemcpyHostToDevice);
    cudaMemcpy((void **)&dev_filters, (void **)&host_filters, sizeof(FiltersKernel), cudaMemcpyHostToDevice);

    //  inite devices pointers
    //  dev_inputData
    size_t size_bytes_devInputData = size_t(host_inputData->rows) * host_inputData->cols * host_inputData->channelStep;
    cudaMallocManaged((void **)(&dev_inputData->data), size_bytes_devInputData);
    cudaMemcpy((void **)(&dev_inputData->data), (void **)(&host_inputData->data), sizeof(size_bytes_devInputData), cudaMemcpyHostToDevice);

    // dev_outputData
    size_t size_bytes_devOutputData = size_t(host_outputData->rows) * host_outputData->cols * host_outputData->channelStep;
    cudaMallocManaged((void **)(&dev_outputData->data), sizeof(size_bytes_devOutputData));
    cudaMemcpy((void **)(&dev_outputData->data), (void **)(&host_outputData->data), size_bytes_devOutputData, cudaMemcpyHostToDevice);

    // dev_filters->weights.data
    size_t size_bytes_devfilters_weightsData = size_t(host_filters->weights.rows) * host_filters->weights.cols * host_filters->weights.channelStep;
    cudaMallocManaged((void **)(&dev_filters->weights.data), size_bytes_devfilters_weightsData);
    cudaMemcpy((void **)(&dev_filters->weights.data), (void **)(&host_filters->weights.data), size_bytes_devfilters_weightsData, cudaMemcpyHostToDevice);

    // dev_filters->biases
    size_t size_bytes_devfilters_biasesData = size_t(host_filters->biases.rows) * host_filters->biases.cols * host_filters->biases.channelStep;
    cudaMallocManaged((void **)(&dev_filters->biases.data), size_bytes_devfilters_biasesData);
    cudaMemcpy((void **)(&dev_filters->biases.data), (void **)(&host_filters->biases.data), size_bytes_devfilters_biasesData, cudaMemcpyHostToDevice);


    // kernnel


    // store the results into the host_outputData
    cudaMemcpy((void **)&host_outputData, (void **)&dev_outputData, sizeof(CDataBlobKernel), cudaMemcpyDeviceToHost);
    cudaMemcpy((void **)(&host_outputData->data), (void **)(&dev_outputData->data), size_bytes_devOutputData, cudaMemcpyDeviceToHost);

    // free the allocated memory
    cudaFree(dev_inputData->data);
    cudaFree(dev_outputData->data);
    cudaFree(dev_filters->weights.data);
    cudaFree(dev_filters->biases.data);

    cudaFree(dev_inputData);
    cudaFree(dev_outputData);
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