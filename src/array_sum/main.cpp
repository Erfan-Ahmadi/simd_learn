#include <iostream>
#include <chrono>
#include <time.h>
#include <immintrin.h>
#include <vector>

constexpr size_t size = 1 << 20;
constexpr size_t n = 50; // Run N-Times

int main()
{
	srand(time(NULL));
	double* first = static_cast<double*>(malloc(size * sizeof(double)));
	double* second = static_cast<double*>(malloc(size * sizeof(double)));

	for (size_t i = 0; i < size; ++i)
	{
		first[i] = rand() % 1024;
		second[i] = rand() % 1024;
	}

	double* out_seq = static_cast<double*>(malloc(size * sizeof(double)));
	double* out_simd = static_cast<double*>(malloc(size * sizeof(double)));

	// Sequential
	double sum_seq = 0;
	for (size_t x = 0; x < n; ++x)
	{
		const auto t1 = std::chrono::high_resolution_clock::now();

		#pragma loop(no_vector)
		for (size_t i = 0; i < size; ++i)
			out_seq[i] = std::sqrt(first[i] * first[i] + second[i] * second[i]);

		const auto t2 = std::chrono::high_resolution_clock::now();
		sum_seq += std::chrono::duration<double, std::milli>(t2 - t1).count();
	}

	// SIMD

	double sum_simd = 0;
	for (size_t x = 0; x < n; ++x)
	{
		const auto t2 = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < size / 4; ++i)
		{
			__m256d first_vec = _mm256_load_pd(&first[i * 4]);
			__m256d second_vec = _mm256_load_pd(&second[i * 4]);

			volatile __m256d result = _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(first_vec, first_vec), _mm256_mul_pd(second_vec, second_vec)));
			//_mm256_store_pd(&out_simd[i * 4], result);
		}

		const auto t3 = std::chrono::high_resolution_clock::now();
		sum_simd += std::chrono::duration<double, std::milli>(t3 - t2).count();
	}

	std::cout << "\n	Avg Sequential: " << sum_seq / n << "(ms)" << std::endl;
	std::cout << "\n	Avg SIMD: " << sum_simd / n << "(ms)" << std::endl;

	// Check Results
	bool equal = true;
	for (size_t i = 0; i < size; ++i)
	{
		if (out_seq[i] != out_simd[i])
		{
			equal = false;
			break;
		}
	}

	std::cout << "\n	Results: " << ((equal) ? "Success" : "Fail") << std::endl;

	free(first);
	free(second);
	free(out_seq);
	free(out_simd);
}