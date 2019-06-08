#include "../common.hpp"

int main()
{
	std::cout << "Hello, Masked Calculate!" << std::endl;

	__m256 first = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
	__m256 second = _mm256_setr_ps(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
	
	__m256 result = _mm256_cmp_ps(first, second, _CMP_EQ_OQ);
	
	float* result_arr = reinterpret_cast<float*>(&result);

	for (size_t i = 0; i < 8; ++i)
		std::cout << i << " = " << result_arr[i] << std::endl;
}