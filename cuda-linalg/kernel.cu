#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
#include <algorithm>

__global__ void addKernel(double* c, const double* a, const double* b, size_t size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (size_t j = 0; j < size; j += blockDim.x)
	{
		if (i + j < size)
			c[i + j] = a[i + j] + b[i + j];
	}
}

__global__ void subKernel(double* c, const double* a, const double* b, size_t size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (size_t j = 0; j < size; j += blockDim.x)
	{
		if (i + j < size)
			c[i + j] = a[i + j] - b[i + j];
	}
}

__global__ void mulKernel(double* c, const double* a, double* b, size_t size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (size_t j = 0; j < size; j += blockDim.x)
	{
		if (i + j < size)
			c[i + j] = a[i + j] * (*b);
	}
}

void CUDA::Vector::add(double* result, const double* left, const double* right, size_t size)
{
	double* dev_left = nullptr;
	double* dev_right = nullptr;
	double* dev_result = nullptr;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	cudaMalloc((void**)&dev_left, size * sizeof(double));
	cudaMalloc((void**)&dev_right, size * sizeof(double));
	cudaMalloc((void**)&dev_result, size * sizeof(double));

	cudaMemcpy(dev_left, left, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_right, right, size * sizeof(double), cudaMemcpyHostToDevice);
	
	addKernel << < std::min(prop.maxGridSize[0], (int)size / prop.maxThreadsPerBlock + 1), std::min((int)size, prop.maxThreadsPerBlock) >> > (dev_result, dev_left, dev_right, size);
	
	cudaMemcpy(result, dev_result, size * sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaFree(dev_left);
	cudaFree(dev_right);
	cudaFree(dev_result);
}

void CUDA::Vector::sub(double* result, const double* left, const double* right, size_t size)
{
	double* dev_left = nullptr;
	double* dev_right = nullptr;
	double* dev_result = nullptr;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	cudaMalloc((void**)&dev_left, size * sizeof(double));
	cudaMalloc((void**)&dev_right, size * sizeof(double));
	cudaMalloc((void**)&dev_result, size * sizeof(double));

	cudaMemcpy(dev_left, left, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_right, right, size * sizeof(double), cudaMemcpyHostToDevice);

	subKernel << < std::min(prop.maxGridSize[0], (int)size / prop.maxThreadsPerBlock + 1), std::min((int)size, prop.maxThreadsPerBlock) >> > (dev_result, dev_left, dev_right, size);

	cudaMemcpy(result, dev_result, size * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(dev_left);
	cudaFree(dev_right);
	cudaFree(dev_result);
}

void CUDA::Vector::mul(double* result, const double* left, double right, size_t size)
{
	double* dev_left = nullptr;
	double* dev_right = nullptr;
	double* dev_result = nullptr;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	cudaMalloc((void**)&dev_left, size * sizeof(double));
	cudaMalloc((void**)&dev_right, sizeof(double));
	cudaMalloc((void**)&dev_result, size * sizeof(double));

	cudaMemcpy(dev_left, left, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_right, &right, sizeof(double), cudaMemcpyHostToDevice);

	mulKernel << < std::min(prop.maxGridSize[0], (int)size / prop.maxThreadsPerBlock + 1), std::min((int)size, prop.maxThreadsPerBlock) >> > (dev_result, dev_left, dev_right, size);

	cudaMemcpy(result, dev_result, size * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(dev_left);
	cudaFree(dev_right);
	cudaFree(dev_result);
}

__global__ void addMat(double** c, double** a, double** b, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	for (size_t k = 0; k < n; k+= blockDim.x)
	{
		for (size_t l = 0; l < n; l += blockDim.y)
		{
			if (i + k < n && j + l < n)
				c[i + k][j + l] = a[i + k][j + l] + b[i + k][j + l];
		}
	}
}

__global__ void subMat(double** c, double** a, double** b, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	for (size_t k = 0; k < n; k += blockDim.x)
	{
		for (size_t l = 0; l < n; l += blockDim.y)
		{
			if (i + k < n && j + l < n)
				c[i + k][j + l] = a[i + k][j + l] - b[i + k][j + l];
		}
	}
}

__global__ void mulMat(double** c, double** a, double** b, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	for (size_t k = 0; k < n; k += blockDim.x)
	{
		for (size_t l = 0; l < n; l += blockDim.y)
		{
			if (i + k < n && j + l < n)
			{
				c[i + k][j + l] = 0;
				for (size_t m = 0; m < n; m++)
				{
					c[i + k][j + l] += a[i + k][m] * b[m][j + l];
				}
			}
		}
	}
}

__global__ void mulMat(double** c, double** a, double *b, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	for (size_t k = 0; k < n; k += blockDim.x)
	{
		for (size_t l = 0; l < n; l += blockDim.y)
		{
			if (i + k < n && j + l < n)
				c[i + k][j + l] = a[i + k][j + l] * (*b);
		}
	}
}

void CUDA::Matrix::add(double** result, double** left, double** right, size_t size)
{
	double** dev_left = nullptr;
	double** dev_right = nullptr;
	double** dev_result = nullptr;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	
	cudaMalloc((void**)&dev_left, size * sizeof(double*));
	cudaMalloc((void**)&dev_right, size * sizeof(double*));
	cudaMalloc((void**)&dev_result, size * sizeof(double*));
	
	for (size_t i = 0; i < size; i++)
	{
		cudaMalloc((void**)&dev_right[i], size * sizeof(double));
		cudaMalloc((void**)&dev_left[i], size * sizeof(double));
		cudaMalloc((void**)&dev_result[i], size * sizeof(double));
		
		cudaMemcpy(dev_left[i], left[i], size * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_right[i], right[i], size * sizeof(double), cudaMemcpyHostToDevice);
	}

	dim3 dimGrid(std::min((int)size, prop.maxGridSize[0]), std::min((int)size, prop.maxGridSize[1]));
	dim3 dimBlock(std::min((int)size, prop.maxThreadsDim[0]), std::min((int)size, prop.maxThreadsDim[1]));

	addMat << < dimGrid, dimBlock >> > (dev_result, dev_left, dev_right, size);
	
	cudaMemcpy(result, dev_result, size * sizeof(double*), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < size; i++)
	{
		cudaFree(dev_left[i]);
		cudaFree(dev_right[i]);
		cudaFree(dev_result[i]);
	}
	cudaFree(dev_left);
	cudaFree(dev_right);
	cudaFree(dev_result);
}

void CUDA::Matrix::sub(double** result, double** left, double** right, size_t size)
{
	double** dev_left = nullptr;
	double** dev_right = nullptr;
	double** dev_result = nullptr;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	cudaMalloc((void**)&dev_left, size * sizeof(double*));
	cudaMalloc((void**)&dev_right, size * sizeof(double*));
	cudaMalloc((void**)&dev_result, size * sizeof(double*));

	for (size_t i = 0; i < size; i++)
	{
		cudaMalloc((void**)&dev_right[i], size * sizeof(double));
		cudaMalloc((void**)&dev_left[i], size * sizeof(double));
		cudaMalloc((void**)&dev_result[i], size * sizeof(double));

		cudaMemcpy(dev_left[i], left[i], size * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_right[i], right[i], size * sizeof(double), cudaMemcpyHostToDevice);
	}

	dim3 dimGrid(std::min((int)size, prop.maxGridSize[0]), std::min((int)size, prop.maxGridSize[1]));
	dim3 dimBlock(std::min((int)size, prop.maxThreadsDim[0]), std::min((int)size, prop.maxThreadsDim[1]));

	subMat << < dimGrid, dimBlock >> > (dev_result, dev_left, dev_right, size);

	cudaMemcpy(result, dev_result, size * sizeof(double*), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < size; i++)
	{
		cudaFree(dev_left[i]);
		cudaFree(dev_right[i]);
		cudaFree(dev_result[i]);
	}
	cudaFree(dev_left);
	cudaFree(dev_right);
	cudaFree(dev_result);
}

void CUDA::Matrix::mul(double** result, double** left, double** right, size_t size)
{
	double** dev_left = nullptr;
	double** dev_right = nullptr;
	double** dev_result = nullptr;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	cudaMalloc((void**)&dev_left, size * sizeof(double*));
	cudaMalloc((void**)&dev_right, size * sizeof(double*));
	cudaMalloc((void**)&dev_result, size * sizeof(double*));

	for (size_t i = 0; i < size; i++)
	{
		cudaMalloc((void**)&dev_right[i], size * sizeof(double));
		cudaMalloc((void**)&dev_left[i], size * sizeof(double));
		cudaMalloc((void**)&dev_result[i], size * sizeof(double));

		cudaMemcpy(dev_left[i], left[i], size * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_right[i], right[i], size * sizeof(double), cudaMemcpyHostToDevice);
	}

	dim3 dimGrid(std::min((int)size, prop.maxGridSize[0]), std::min((int)size, prop.maxGridSize[1]));
	dim3 dimBlock(std::min((int)size, prop.maxThreadsDim[0]), std::min((int)size, prop.maxThreadsDim[1]));

	mulMat << < dimGrid, dimBlock >> > (dev_result, dev_left, dev_right, size);

	cudaMemcpy(result, dev_result, size * sizeof(double*), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < size; i++)
	{
		cudaFree(dev_left[i]);
		cudaFree(dev_right[i]);
		cudaFree(dev_result[i]);
	}
	cudaFree(dev_left);
	cudaFree(dev_right);
	cudaFree(dev_result);
}

void CUDA::Matrix::mul(double** result, double** left, double right, size_t size)
{
	double** dev_left = nullptr;
	double** dev_result = nullptr;
	double* dev_right;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	cudaMalloc((void**)&dev_left, size * sizeof(double*));
	cudaMalloc((void**)&dev_result, size * sizeof(double*));
	cudaMalloc((void**)&dev_right, sizeof(double));

	cudaMemcpy(dev_right, &right, sizeof(double), cudaMemcpyHostToDevice);

	for (size_t i = 0; i < size; i++)
	{
		cudaMalloc((void**)&dev_left[i], size * sizeof(double));
		cudaMalloc((void**)&dev_result[i], size * sizeof(double));

		cudaMemcpy(dev_left[i], left[i], size * sizeof(double), cudaMemcpyHostToDevice);
	}

	dim3 dimGrid(std::min((int)size, prop.maxGridSize[0]), std::min((int)size, prop.maxGridSize[1]));
	dim3 dimBlock(std::min((int)size, prop.maxThreadsDim[0]), std::min((int)size, prop.maxThreadsDim[1]));

	mulMat << < dimGrid, dimBlock >> > (dev_result, dev_left, dev_right, size);

	cudaMemcpy(result, dev_result, size * sizeof(double*), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < size; i++)
	{
		cudaFree(dev_left[i]);
		cudaFree(dev_result[i]);
	}
	cudaFree(dev_left);
	cudaFree(dev_result);
	cudaFree(dev_right);
}