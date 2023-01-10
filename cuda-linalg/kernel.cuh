#pragma once

namespace CUDA {
	namespace Vector {
		void add(double* result, const double* left, const double* right, size_t size);

		void sub(double* result, const double* left, const double* right, size_t size);

		void mul(double* result, const double* left, double right, size_t size);
	};
	namespace Matrix {
		void add(double** result, double** left, double** right, size_t size);

		void sub(double** result, double** left, double** right, size_t size);

		void mul(double** result, double** left, double** right, size_t size);

		void mul(double** result, double** left, double right, size_t size);
	};
};