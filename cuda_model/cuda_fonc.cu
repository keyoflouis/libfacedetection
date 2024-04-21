#include "cuda_fonc.cuh"

#include <stdio.h>

#define thread_per_block 256
__global__ void kernel(CDataBlobKernel* inputData, CDataBlobKernel* outputData, FiltersKernel* filters) {
	
	//size_t size_bytes_devfilters_weightsData = size_t(filters->weights.rows) * filters->weights.cols * filters->weights.channelStep;
	//for (int i = 0; i < size_bytes_devfilters_weightsData /4; i++) {
	//	printf("%f \n", filters->weights.data[i]);
	//}

	/*size_t size_bytes_devInputData = size_t(inputData->rows) * inputData->cols * inputData->channelStep;
	for (int i = 0; i < size_bytes_devInputData;i++) {
		printf("%f \n", inputData->data[i]);
	}*/

	/*size_t size_bytes_devOutputData = size_t(outputData->rows) * outputData->cols * outputData->channelStep;
	for (int i = 0; i < size_bytes_devOutputData;i++) {
		printf("%f \n", outputData->data[i]);
	}*/

	/*atomicAdd(&outputData->data[1], 1.0f);*/
	/*outputData->data[1] = 1.0f;
	printf("%f", outputData->data[1]);*/
	
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
					pOut[ch] = sum;
					pOut[ch] += filters->biases.data[ch];
					/*if (!(ch % 100))
						printf("%f \n", pOut[ch]);*/
				}
					
			}
			
	
	  //for (int row = 0; row < outputData->rows; row++)
   // {
   //     for (int col = 0; col < outputData->cols; col++)
   //     {
   //         float *pOut = outputData->ptr(row, col);
   //         const float *pIn = inputData->ptr(row, col);

   //         for (int ch = 0; ch < outputData->channels; ch++)
   //         {
   //             const float *pF = filters->weights.ptr(0, ch);
   //             float sum = 0.f;
   //             for (int i = 0; i < inputData->channels; i++)
   //             {
   //                 sum += (pIn[i] * pF[i]);
   //             }
   //             pOut[ch] = sum;
   //             pOut[ch] += filters->biases.data[ch];
			//		
			//	//if (!(ch % 100))
			//	//	printf("%f \n", pOut[ch]);

   //         }
   //     }
   // }
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


	
	
	CDataBlobKernel* dev_inputData;
	CDataBlobKernel* dev_outputData;
	FiltersKernel* dev_filters;


	cudaMalloc((void**)&dev_inputData,sizeof(CDataBlobKernel));
	cudaMalloc((void**)&dev_outputData,sizeof(CDataBlobKernel));
	cudaMalloc((void**)&dev_filters,sizeof(FiltersKernel));
	
	
	float* dev_input_data;
	float* dev_output_data;

	// allocate copy and deep copy for inpute
 	size_t size_bytes_devInputData = size_t(inputData.rows) * inputData.cols * inputData.channelStep;
	cudaMemcpy(dev_inputData,&inputData,sizeof(CDataBlobKernel),cudaMemcpyHostToDevice);


	// deep copy for inpute
	cudaMalloc((void**)&dev_input_data, size_bytes_devInputData);
	cudaMemcpy(dev_input_data ,inputData.data, size_bytes_devInputData,cudaMemcpyHostToDevice);
	cudaMemcpy( &dev_inputData->data , &dev_input_data , sizeof(float*) , cudaMemcpyHostToDevice );

	// allocate copy and deep copy for outpute
	size_t size_bytes_devOutputData = size_t(outputData.rows) * outputData.cols * outputData.channelStep;
	cudaMemcpy(dev_outputData, &outputData, sizeof(CDataBlobKernel), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_output_data , size_bytes_devOutputData );
	cudaMemcpy(dev_output_data, outputData.data, size_bytes_devOutputData, cudaMemcpyHostToDevice);
	cudaMemcpy( &dev_outputData->data , &dev_output_data , sizeof(float*) , cudaMemcpyHostToDevice );


	/*for (int i = 0; i < size_bytes_devInputData / 4; i++) {
		printf("cpu :%d", inputData.data[i]);
	}*/

	
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

	kernel << <dimGrid, dimBlock >> > (dev_inputData, dev_outputData, dev_filters);

	cudaDeviceSynchronize();
	// store the results
	void* temp = outputData.data;
	cudaMemcpy(&outputData , dev_outputData,sizeof(CDataBlobKernel),cudaMemcpyDeviceToHost);
	outputData.data = (float*)temp;
	cudaMemcpy(outputData.data, dev_output_data, size_bytes_devOutputData, cudaMemcpyDeviceToHost);


	// free
	
	cudaFree(dev_input_data);
	cudaFree(dev_inputData);

	cudaFree(dev_output_data);
	cudaFree(dev_outputData);

	cudaFree(dev_filters_weightsData);
	cudaFree(dev_filters_biasesData);
	cudaFree(dev_filters);
	
	//printf("%f", outputData.data[1]);

  //for (int row = 0; row < outputData.rows; row++)
  //  {
  //      for (int col = 0; col < outputData.cols; col++)
  //      {
  //          float *pOut = outputData.ptr(row, col);
  //          const float *pIn = inputData.ptr(row, col);

  //          for (int ch = 0; ch < outputData.channels; ch++)
  //          {
  //              const float *pF = filters.weights.ptr(0, ch);
  //              float sum = 0.f;
  //              for (int i = 0; i < inputData.channels; i++)
  //              {
  //                  sum += (pIn[i] * pF[i]);
  //              }
  //              pOut[ch] = sum;
  //              pOut[ch] += filters.biases.data[ch];
  //          }
  //      }
  //  }
    return &outputData;
};