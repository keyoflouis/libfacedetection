#include"cuda_fonc.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>

__global__ void sub_dotproduct(){
    printf("hello cuda!");
}

void product(){
    sub_dotproduct<<<1, 1>>>();
}