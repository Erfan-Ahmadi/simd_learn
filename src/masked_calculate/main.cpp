#include "../common.hpp"
// use mask_store with AVX-512

// Code From https://stackoverflow.com/questions/36932240
inline __m256 pack_left_256(const __m256& src, const unsigned int& mask)
{
	uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
	expanded_mask *= 0xFF;

	const uint64_t identity_indices = 0x0706050403020100;
	uint64_t wanted_indices = _pext_u64(identity_indices, expanded_mask);

	__m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
	__m256i shufmask = _mm256_cvtepu8_epi32(bytevec);

	print_vec_256<__m256i, uint32_t>(shufmask);

	return _mm256_permutevar8x32_ps(src, shufmask);
}

int main()
{
	std::cout << "Hello, Masked Calculate!" << std::endl;

	// Initialize Data
	__m256 first = _mm256_setr_ps(0.0, -1.0, 2.0, 3.0, 4.0, 6.0, -2.0, 5.0);
	__m256 limit = _mm256_set1_ps(1.0);

	// Compare
	__m256 result = _mm256_cmp_ps(first, limit, _CMP_GE_OQ);
	
	// Get Mask From Compare Result (IMPORTANT)
	int mask = _mm256_movemask_ps(result);

	// Pack Left with Mask
	__m256 packed = pack_left_256(first, mask);
	
	// Store Results
	float results[8];
	_mm256_store_ps(results, packed);

	for(short i = 0; i < 8; ++i)
		std::cout << results[i] << std::endl;
}