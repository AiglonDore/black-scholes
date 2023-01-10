#include "vector.h"
#include <cstring>
#include <utility>

#include <cuda_runtime.h>
#include "kernel.cuh"

Vector::Vector()
{
	data = nullptr;
	size = 0;
	int nbDevices(0);
	cudaGetDeviceCount(&nbDevices);
	GPU = nbDevices != 0;
}

Vector::Vector(const Vector& other) noexcept
{
	size = other.size;
	GPU = other.GPU;
	data = new double[size];
	memcpy(data, other.data, size * sizeof(double));
}

Vector::Vector(Vector&& other) noexcept
{
	size = std::move(other.size);
	GPU = std::move(other.GPU);
	data = other.data;
	other.data = nullptr;
}

Vector::Vector(size_t size, bool GPU) : size(size), GPU(GPU)
{
	int nbDevices(0);
	cudaGetDeviceCount(&nbDevices);
	GPU = nbDevices != 0;
	data = new double[size];
}

Vector::Vector(size_t size, double* data, bool GPU) : size(size), GPU(GPU)
{
	int nbDevices(0);
	cudaGetDeviceCount(&nbDevices);
	GPU = nbDevices != 0;
	this->data = new double[size];
	memcpy(this->data, data, size * sizeof(double));
}

Vector::Vector(size_t size, const double* data, bool GPU) : size(size), GPU(GPU)
{
	int nbDevices(0);
	cudaGetDeviceCount(&nbDevices);
	GPU = nbDevices != 0;
	this->data = new double[size];
	memcpy(this->data, data, size * sizeof(double));
}

Vector::~Vector()
{
	delete[] data;
}

Vector::Vector(const std::vector<double>& data, bool GPU) : size(data.size()), GPU(GPU)
{
	int nbDevices(0);
	cudaGetDeviceCount(&nbDevices);
	GPU = nbDevices != 0;
	this->data = new double[size];
	memcpy(this->data, data.data(), size * sizeof(double));
}

Vector& Vector::operator=(const Vector& other) noexcept
{
	delete[] data;
	size = other.size;
	GPU = other.GPU;
	data = new double[size];
	memcpy(data, other.data, size * sizeof(double));
	return *this;
}

Vector& Vector::operator=(Vector&& other) noexcept
{
	delete[] data;
	size = std::move(other.size);
	GPU = std::move(other.GPU);
	data = other.data;
	other.data = nullptr;
	return *this;
}

bool Vector::operator==(const Vector& other) const
{
	if (this->size != other.size) return false;
	for (size_t i = 0; i < size; i++)
	{
		if (data[i] != other.data[i]) return false;
	}return true;
}

Vector& Vector::operator+=(const Vector& other)
{
	if (this->size != other.size) throw std::runtime_error("Vectors must have the same size");
	if (GPU && other.GPU)
	{
		CUDA::Vector::add(this->data, this->data, other.data, size);
	}
	else
	{
		for (size_t i = 0; i < size; i++)
		{
			this->data[i] += other.data[i];
		}
	}
	return *this;
}

Vector& Vector::operator-=(const Vector& other)
{
	if (this->size != other.size) throw std::runtime_error("Vectors must have the same size");
	if (GPU && other.GPU)
	{

	}
	else
	{
		for (size_t i = 0; i < size; i++)
		{
			this->data[i] -= other.data[i];
		}
	}
	return *this;
}

Vector& Vector::operator*=(double other)
{
	if (GPU)
	{

	}
	else
	{
		for (size_t i = 0; i < size; i++)
		{
			this->data[i] *= other;
		}
	}
	return *this;
}

Vector operator+(const Vector& left, const Vector& right)
{
	return Vector(left) += right;
}

Vector operator-(const Vector& left, const Vector& right)
{
	return Vector(left) -= right;
}

Vector operator*(const Vector& left, double scalar)
{
	return Vector(left) *= scalar;
}
