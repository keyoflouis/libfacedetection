#include"cuda_fonc.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>

//#define thread_per_block 256
//__global__ void sub_dotproduct(float* dev_p1,float* dev_p2,float* dev_sum,int num){
//    __shared__ float cache[thread_per_block];
//
//    int tid = threadIdx.x + blockDim.x * blockIdx.x;
//    int cacheIndex = threadIdx.x;
//    float temp = 0;
//
//    while (tid < num)
//    {
//        temp += dev_p1[tid] * dev_p2[tid];
//        tid += gridDim.x * blockDim.x;
//    }
//
//    cache[cacheIndex] = temp;
//    __syncthreads();
//
//    int i = blockDim.x / 2;
//    while (i != 0)
//    {   
//        if(cacheIndex<i)
//            cache[cacheIndex] += cache[i + cacheIndex];
//        __syncthreads();
//        i /= 2;
//    }
//
//    if (cacheIndex == 0) {
//        atomicAdd(dev_sum, cache[0]);
//    }
//}

//void cuda_dotProduct(const float* p1,const float* p2,float& sum,int num){
//    float* dev_p1 =nullptr;
//    float* dev_p2 =nullptr;
//    float* dev_sum=nullptr;
//
//    cudaMalloc((void**)&dev_p1, num * sizeof(float));
//    cudaMalloc((void**)&dev_p2, num * sizeof(float));
//    cudaMalloc((void**)&dev_sum, sizeof(float));
//
//    cudaMemcpy(dev_p1, p1, num * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(dev_p2, p2, num * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemset(dev_sum, 0, sizeof(float));
//
//    sub_dotproduct << <(num + thread_per_block - 1) / thread_per_block, thread_per_block >> > (dev_p1, dev_p2, dev_sum, num);
//
//    cudaMemcpy(&sum, dev_sum,sizeof(float), cudaMemcpyDeviceToHost);
//
//    cudaFree(dev_p1);
//    cudaFree(dev_p2);
//    cudaFree(dev_sum);
//}


__global__ void dotProductKernel(float* result, const float* p1, const float* p2, const int num) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float localSum = 0.0f;

    for (int i = index; i < num; i += blockDim.x * gridDim.x) {
        localSum += p1[i] * p2[i];
    }

    atomicAdd(result, localSum);
}

DotProductGPU::~DotProductGPU() {
    cudaFree(d_sum);
    cudaFree(d_vec1);
    cudaFree(d_vec2);
}

DotProductGPU::DotProductGPU(const int num_) :num(num_) {
    cudaMalloc((float**)&d_sum, 1 * sizeof(float));
    cudaMalloc((float**)&d_vec1, num * sizeof(float));
    cudaMalloc((float**)&d_vec2, num * sizeof(float));
    numBlocks = (num + blockSize - 1) / blockSize;
}

void DotProductGPU::operator()(float& sum, const float* p1, const float* p2) {
    cudaMemcpy(d_sum, &sum, 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec1, p1, num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, p2, num * sizeof(float), cudaMemcpyHostToDevice);
    dotProductKernel << <numBlocks, blockSize >> > (d_sum, d_vec1, d_vec2, num);
    cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
}

//
//void dotProductGPU(float& sum, const float* p1, const float* p2, const int num) {
//    float* d_sum;
//    float* d_vec1;
//    float* d_vec2;
//
//    cudaMalloc((float**)&d_sum, 1 * sizeof(float));
//    cudaMalloc((float**)&d_vec1, num * sizeof(float));
//    cudaMalloc((float**)&d_vec2, num * sizeof(float));
//
//    cudaMemcpy(d_sum, &sum, 1 * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_vec1, p1, num * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_vec2, p2, num * sizeof(float), cudaMemcpyHostToDevice);
//
//    int blockSize = 256;
//    int numBlocks = (num + blockSize - 1) / blockSize;
//    dotProductKernel << <numBlocks, blockSize >> > (d_sum, d_vec1, d_vec2, num);
//    //sub_dotproduct << <numBlocks, blockSize >> > (d_vec1, d_vec2, d_sum, num);
//
//    cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
//    cudaFree(d_sum);
//    cudaFree(d_vec1);
//    cudaFree(d_vec2);
//}