#pragma once

namespace CUDA {
	namespace Vector {
		void add(double* result, const double* left, const double* right, size_t size);
	}
};