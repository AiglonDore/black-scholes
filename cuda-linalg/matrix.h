#pragma once

#include <vector>
#include <exception>
#include <stdexcept>

class Matrix
{
private:
	double** data;
	size_t size;
	bool GPU;
public:
	Matrix();
	Matrix(const Matrix& other) noexcept;
	Matrix(Matrix&& other) noexcept;

	explicit Matrix(size_t size, bool GPU = true);
	explicit Matrix(size_t size, double** data, bool GPU = true);
	explicit Matrix(size_t size, const double** data, bool GPU = true);
	explicit Matrix(const std::vector<std::vector<double>>& data, bool GPU = true);

	~Matrix();

	Matrix& operator=(const Matrix& other) noexcept;
	Matrix& operator=(Matrix&& other) noexcept;

	inline double* operator[](size_t index) { return data[index]; };
	inline const double* operator[](size_t index) const { return data[index]; };

	inline double& at(size_t index, size_t index2) { return data[index][index2]; };
	inline const double& at(size_t index, size_t index2) const { return data[index][index2]; };

	inline double** getData() { return data; };

	inline size_t getSize() const { return size; };
	inline bool isGPUEnabled() const { return GPU; };

	void setGPU(bool GPU) { this->GPU = GPU; };

	bool operator==(const Matrix& other) const;
	bool operator!=(const Matrix& other) const { return !(*this == other); };

	Matrix& operator+=(const Matrix& other);
	Matrix& operator-=(const Matrix& other);
	Matrix& operator*=(const Matrix& other);
	Matrix& operator*=(double scalar);
	Matrix& operator/=(double scalar) { return *this *= 1.0 / scalar; };

	void toVector(std::vector<std::vector<double>>& data) const;
};

Matrix operator+(const Matrix& left, const Matrix& right);
Matrix operator-(const Matrix& left, const Matrix& right);
Matrix operator*(const Matrix& left, const Matrix& right);
Matrix operator*(const Matrix& left, double scalar);
Matrix operator*(double scalar, const Matrix& right) { return right * scalar; };
Matrix operator/(const Matrix& left, double scalar) { return left * (1 / scalar); };