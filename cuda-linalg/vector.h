#pragma once

#include <vector>
#include <exception>
#include <stdexcept>

class Vector
{
private:
	double* data;
	size_t size;
	bool GPU;
public:
	Vector();
	Vector(const Vector& other) noexcept;
	Vector(Vector&& other) noexcept;
	
	explicit Vector(size_t size, bool GPU = true);
	explicit Vector(size_t size, double* data, bool GPU = true);
	explicit Vector(size_t size, const double* data, bool GPU = true);
	explicit Vector(const std::vector<double>& data, bool GPU = true);
	
	~Vector();

	Vector& operator=(const Vector& other) noexcept;
	Vector& operator=(Vector&& other) noexcept;

	inline double& operator[](size_t index) { return data[index]; };
	inline const double& operator[](size_t index) const { return data[index]; };

	inline double& at(size_t index) { return data[index]; };
	inline const double& at(size_t index) const { return data[index]; };

	inline double* getData() { return data; };
	inline const double* getData() const { return data; };

	inline size_t getSize() const { return size; };
	inline bool isGPUEnabled() const { return GPU; };

	void setGPU(bool GPU) { this->GPU = GPU; };
	
	bool operator==(const Vector& other) const;
	bool operator!=(const Vector& other) const { return !(*this == other); };

	Vector& operator+=(const Vector& other);
	Vector& operator-=(const Vector& other);
	Vector& operator*=(double scalar);
	Vector& operator/=(double scalar) { return *this *= 1.0 / scalar; };

	void toVector(std::vector<double>& data) const;
};

Vector operator+(const Vector& left, const Vector& right);
Vector operator-(const Vector& left, const Vector& right);
Vector operator*(const Vector& left, double scalar);
Vector operator*(double scalar, const Vector& right) { return right * scalar; };
Vector operator/(const Vector& left, double scalar) { return left * (1 / scalar); };