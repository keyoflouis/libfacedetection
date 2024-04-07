#include"cuda_fonc.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>

#define thread_per_block 256
__global__ void sub_dotproduct(float* dev_p1,float* dev_p2,float* dev_sum,int num){
    __shared__ float cache[thread_per_block];

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;

    while (tid < num)
    {
        temp += dev_p1[tid] * dev_p2[tid];
        tid += gridDim.x * blockDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {   
        if(cacheIndex<i)
            cache[cacheIndex] += cache[i + cacheIndex];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        atomicAdd(dev_sum, cache[0]);
    }
}

void cuda_dotProduct(const float* p1,const float* p2,float& sum,int num){
    float* dev_p1 =nullptr;
    float* dev_p2 =nullptr;
    float* dev_sum=nullptr;

    cudaMalloc((void**)&dev_p1, num * sizeof(float));
    cudaMalloc((void**)&dev_p2, num * sizeof(float));
    cudaMalloc((void**)&dev_sum, sizeof(float));

    cudaMemcpy(dev_p1, p1, num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p2, p2, num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dev_sum, 0, sizeof(float));

    sub_dotproduct << <(num + thread_per_block - 1) / thread_per_block, thread_per_block >> > (dev_p1, dev_p2, dev_sum, num);

    cudaMemcpy(&sum, dev_sum,sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_p1);
    cudaFree(dev_p2);
    cudaFree(dev_sum);
}