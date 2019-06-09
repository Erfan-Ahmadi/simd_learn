#include "../common.hpp"

// use mask_store with AVX-512

__m256 compress256(__m256 src, unsigned int mask)
{
  uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);  // unpack each bit to a byte
  expanded_mask *= 0xFF;    // mask |= mask<<1 | mask<<2 | ... | mask<<7;
  // ABC... -> AAAAAAAABBBBBBBBCCCCCCCC...: replicate each bit to fill its byte

  const uint64_t identity_indices = 0x0706050403020100;    // the identity shuffle for vpermps, packed to one index per byte
  uint64_t wanted_indices = _pext_u64(identity_indices, expanded_mask);

  __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
  __m256i shufmask = _mm256_cvtepu8_epi32(bytevec);

  return _mm256_permutevar8x32_ps(src, shufmask);
}

int main()
{
	std::cout << "Hello, Masked Calculate!" << std::endl;

	__m256 first = _mm256_setr_ps(0.0, -1.0, 2.0, 3.0, 4.0, 6.0, -2.0, 5.0);
	__m256 limit = _mm256_set1_ps(1.0);

	// Compare
	__m256 result = _mm256_cmp_ps(first, limit, _CMP_GE_OQ);
	
	// Get Mask From Compare Result (IMPORTANT)
	int mask = _mm256_movemask_ps(result);
	__m256i shfl = _mm256_setr_epi32(2, 3, 4, 5, 7, 0, 1, 6);
	int r = _mm_popcnt_u32(mask);

	// Permute with Mask to Align Left
	__m256 packed = compress256(first, mask);
	//__m256 packed = _mm256_permutevar8x32_ps(first, shfl);

	float results[8];
	_mm256_store_ps(results, packed);

	for(short i = 0; i < 8; ++i)
		std::cout << results[i] << std::endl;
}