#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"

__global__ void addKernel(double* c, const double* a, const double* b, size_t size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
		c[i] = a[i] + b[i];
}

void CUDA::Vector::add(double* result, const double* left, const double* right, size_t size)
{
	double* dev_left = nullptr;
	double* dev_right = nullptr;
	double* dev_result = nullptr;

	cudaMalloc((void**)&dev_left, size * sizeof(double));
	cudaMalloc((void**)&dev_right, size * sizeof(double));
	cudaMalloc((void**)&dev_result, size * sizeof(double));

	cudaMemcpy(dev_left, left, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_right, right, size * sizeof(double), cudaMemcpyHostToDevice);
	
	//addKernel << <(size + 255) / 256, 256 >> > (dev_result, dev_left, dev_right, size);
	
	cudaMemcpy(result, dev_result, size * sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaFree(dev_left);
	cudaFree(dev_right);
	cudaFree(dev_result);
}