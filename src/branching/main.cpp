#include "../common.hpp"
#include <limits>
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

	return _mm256_permutevar8x32_ps(src, shufmask);
}

// 4x4 Latin Square
__m128i rows[4] =
{
_mm_setr_epi32(0, 1, 2, 3),
_mm_setr_epi32(1, 2, 3, 0),
_mm_setr_epi32(2, 3, 0, 1),
_mm_setr_epi32(3, 0, 1, 2)
};

__m128i self_check_rows[2] =
{
_mm_setr_epi32(1, 2, 3, 0),
_mm_setr_epi32(2, 3, 1, 0),
};

__m128i reverse_self_check_rows[2] =
{
_mm_setr_epi32(3, 0, 1, 2),
_mm_setr_epi32(3, 2, 0, 1),
};

// multiply by compare result (last self check doesn't fot to 4 bytes
__m128 last_self_check_mask = _mm_setr_ps(1, 1, 0, 0);

static __m128 half_vec = _mm_setr_ps(0.5, 0.5, 0.5, 0.5);
static __m128 minus_vec = _mm_set1_ps(-1);

int main()
{
	std::cout << "Hello, Branching!\n" << std::endl;

	const size_t count = 3;
	__m128 positions[count] =
	{
		_mm_setr_ps(0.0, 0.0, 0.0, 0.0),
		_mm_setr_ps(3.0, 2.0, 4.0, 1.0),
		_mm_setr_ps(2.0, 6.0, 6.0, 0.0)
	};

	for (size_t i = 0; i < count; ++i)
	{
		for (size_t j = 0; j < 2; ++j)
		{
			const __m128 shuffled = _mm_permutevar_ps(positions[i], self_check_rows[j]);

			std::cout << "PREDICT!!";
			__m128 cmp = _mm_min_ps(_mm_cmp_ps(shuffled, positions[i], _CMP_EQ_OQ), half_vec);
			__m128 add_a = cmp;
			__m128 add_b = _mm_permutevar_ps(add_a, reverse_self_check_rows[j]);
			positions[i] = _mm_sub_ps(positions[i], add_b);
			positions[i] = _mm_add_ps(positions[i], add_a);

		}

		// Last Self Check
		{
			const __m128 shuffled = _mm_div_ps(_mm_permutevar_ps(positions[i], self_check_rows[1]), last_self_check_mask);

			__m128 cmp = _mm_min_ps(_mm_cmp_ps(shuffled, positions[i], _CMP_EQ_OQ), half_vec);
			__m128 add_a = cmp;
			__m128 add_b = _mm_permutevar_ps(add_a, reverse_self_check_rows[1]);
			positions[i] = _mm_sub_ps(positions[i], add_b);
			positions[i] = _mm_add_ps(positions[i], add_a);
		}

		for (size_t k = i + 1; k < count; ++k)
		{
			for (size_t j = 0; j < 4; ++j)
			{
				const __m128 shuffled = _mm_permutevar_ps(positions[i], rows[j]);

				__m128 cmp = _mm_min_ps(_mm_cmp_ps(shuffled, positions[k], _CMP_EQ_OQ), half_vec);
				__m128 add_a = cmp;
				__m128 add_b = _mm_permutevar_ps(add_a, rows[j]);
				positions[i] = _mm_sub_ps(positions[i], add_b);
				positions[k] = _mm_add_ps(positions[k], add_a);
			}
		}
	}

	for (size_t i = 0; i < count; ++i)
	{
		print_vec_128<__m128, float>(positions[i]);
		std::cout << std::endl;
	}
}