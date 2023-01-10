#include "matrix.h"
#include "cuda_runtime.h"
#include "kernel.cuh"

#include <exception>
#include <stdexcept>

Matrix::Matrix() : data(nullptr), size(size), GPU(GPU)
{
	int nbDevices(0);
	cudaGetDeviceCount(&nbDevices);
	GPU = nbDevices != 0;
}

Matrix::Matrix(const Matrix& other) noexcept
{
	size = other.size;
	GPU = other.GPU;
	data = new double* [size];
	for (size_t i = 0; i < size; i++)
	{
		data[i] = new double[size];
		for (size_t j = 0; j < size; j++)
			data[i][j] = other.data[i][j];
	}
}

Matrix::Matrix(Matrix&& other) noexcept
{
	size = std::move(other.size);
	GPU = std::move(other.GPU);
	data = other.data;
	other.data = nullptr;
}

Matrix::Matrix(size_t size, bool GPU) : size(size), GPU(GPU)
{
	int nbDevices(0);
	cudaGetDeviceCount(&nbDevices);
	GPU = nbDevices != 0;
	data = new double* [size];
	for (size_t i = 0; i < size; i++)
		data[i] = new double[size];
}

Matrix::Matrix(size_t size, double** data, bool GPU) : size(size), GPU(GPU)
{
	int nbDevices(0);
	cudaGetDeviceCount(&nbDevices);
	GPU = nbDevices != 0;
	this->data = new double* [size];
	for (size_t i = 0; i < size; i++)
	{
		this->data[i] = new double[size];
		for (size_t j = 0; j < size; j++)
			this->data[i][j] = data[i][j];
	}
}

Matrix::Matrix(size_t size, const double** data, bool GPU) : size(size), GPU(GPU)
{
	int nbDevices(0);
	cudaGetDeviceCount(&nbDevices);
	GPU = nbDevices != 0;
	this->data = new double* [size];
	for (size_t i = 0; i < size; i++)
	{
		this->data[i] = new double[size];
		for (size_t j = 0; j < size; j++)
			this->data[i][j] = data[i][j];
	}
}

Matrix::Matrix(const std::vector<std::vector<double>>& data, bool GPU) : size(data.size()), GPU(GPU)
{
	int nbDevices(0);
	cudaGetDeviceCount(&nbDevices);
	GPU = nbDevices != 0;
	this->data = new double* [size];
	for (size_t i = 0; i < size; i++)
	{
		this->data[i] = new double[size];
		for (size_t j = 0; j < size; j++)
			this->data[i][j] = data[i][j];
	}
}

Matrix::~Matrix()
{
	if (data != nullptr)
	{
		for (size_t i = 0; i < size; i++)
			delete[] data[i];
		delete[] data;
	}
}

Matrix& Matrix::operator=(const Matrix& other) noexcept
{
	if (this != &other)
	{
		if (data != nullptr)
		{
			for (size_t i = 0; i < size; i++)
				delete[] data[i];
			delete[] data;
		}
		size = other.size;
		GPU = other.GPU;
		data = new double* [size];
		for (size_t i = 0; i < size; i++)
		{
			data[i] = new double[size];
			for (size_t j = 0; j < size; j++)
				data[i][j] = other.data[i][j];
		}
	}
	return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept
{
	if (this != &other)
	{
		if (data != nullptr)
		{
			for (size_t i = 0; i < size; i++)
				delete[] data[i];
			delete[] data;
		}
		size = std::move(other.size);
		GPU = std::move(other.GPU);
		data = other.data;
		other.data = nullptr;
	}
	return *this;
}

bool Matrix::operator==(const Matrix& other) const
{
	if (size != other.size)
		return false;
	for (size_t i = 0; i < size; i++)
		for (size_t j = 0; j < size; j++)
			if (data[i][j] != other.data[i][j])
				return false;
	return true;
}

Matrix& Matrix::operator+=(const Matrix& other)
{
	if (size != other.size)
		throw std::invalid_argument("Matrix size mismatch");
	if (GPU && other.GPU)
	{
		CUDA::Matrix::add(data, data, other.data, size);
	}
	else
	{
		for (size_t i = 0; i < size; i++)
			for (size_t j = 0; j < size; j++)
				data[i][j] += other.data[i][j];
	}
	return *this;
}

Matrix& Matrix::operator-=(const Matrix& other)
{
	if (size != other.size)
		throw std::invalid_argument("Matrix size mismatch");
	if (GPU && other.GPU)
	{
		CUDA::Matrix::sub(data, data, other.data, size);
	}
	else
	{
		for (size_t i = 0; i < size; i++)
			for (size_t j = 0; j < size; j++)
				data[i][j] -= other.data[i][j];
	}
	return *this;
}

Matrix& Matrix::operator*=(const Matrix& other)
{
	if (size != other.size)
		throw std::invalid_argument("Matrix size mismatch");
	if (GPU && other.GPU)
	{
		CUDA::Matrix::mul(this->data, this->data, other.data, size);
	}
	else
	{
		double** newData = new double* [size];
		for (size_t i = 0; i < size; i++)
		{
			newData[i] = new double[size];
			for (size_t j = 0; j < size; j++)
			{
				newData[i][j] = 0;
				for (size_t k = 0; k < size; k++)
					newData[i][j] += data[i][k] * other.data[k][j];
			}
		}
		for (size_t i = 0; i < size; i++)
			delete[] data[i];
		delete[] data;
		data = newData;
	}
	return *this;
}

Matrix& Matrix::operator*=(double scalar)
{
	if (GPU)
	{
		CUDA::Matrix::mul(data, data, scalar, size);
	}
	else
	{
		for (size_t i = 0; i < size; i++)
			for (size_t j = 0; j < size; j++)
				data[i][j] *= scalar;
	}
	return *this;
}

void Matrix::toVector(std::vector<std::vector<double>>& data) const
{
	data.clear();
	data.resize(size);
	for (size_t i = 0; i < size; i++)
	{
		data[i].resize(size);
		for (size_t j = 0; j < size; j++)
			data[i][j] = this->data[i][j];
	}
}

Matrix operator+(const Matrix& left, const Matrix& right)
{
	Matrix result(left);
	result += right;
	return result;
}

Matrix operator-(const Matrix& left, const Matrix& right)
{
	Matrix result(left);
	result -= right;
	return result;
}

Matrix operator*(const Matrix& left, const Matrix& right)
{
	Matrix result(left);
	result *= right;
	return result;
}

Matrix operator*(const Matrix& left, double scalar)
{
	Matrix result(left);
	result *= scalar;
	return result;
}