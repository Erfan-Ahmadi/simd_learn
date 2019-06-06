#include <iostream>
#include <immintrin.h>
#include <vector>

constexpr size_t instance_count = 1 << 5;
constexpr size_t cols_count = 16;
static_assert(cols_count <= instance_count * 2, "collision count larger than it should be.");
typedef int32_t size;

struct vec2
{
	float x;
	float y;
};

struct samples
{
	std::vector<vec2>	positions;
	std::vector<vec2>	velocities;
	std::vector<float>	scales;

	void resize(const size& size)
	{
		positions.resize(size);
		velocities.resize(size);
		scales.resize(size);
	}

} circles;

static std::vector<size> cols;

inline void initialize_data()
{
	circles.resize(instance_count);
	cols.resize(cols_count * 2);

	for (size i = 0; i < instance_count; ++i)
	{
		circles.positions[i] =
		{
			static_cast<float>(i + 10),
			static_cast<float>(i * 2)
		};

		circles.velocities[i] =
		{
			i * 2.0f,
			i * 2.0f
		};

		circles.scales[i] = 10;
	}

	for (size_t i = 0; i < cols_count * 2; i += 2)
	{
		cols[i] = i % instance_count;
		cols[i + 1] = (i + 1) % instance_count;
	}
}

template <class V, class T>
inline void print_vec(V& vec)
{
	T* vecf = reinterpret_cast<T*>(&vec);

	for (short i = 0; i < 8; ++i)
		std::cout << "(" << i << "): " << vecf[i] << std::endl;

	std::cout << std::endl;
}

int main()
{
	// Single Instruction Multiple Data
	std::cout << "Hello, SIMD Collision!" << std::endl;

	initialize_data();

	// We assume collisions are recorded in cols vector, no collision detection is done!

	// With 256 bit Vector Instructions (8 floats), we should loop (cols_count / 8) times 

	const auto left = (cols_count % 8);
	const auto n = cols_count - left;

	__m256i base = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	__m256i two = _mm256_set1_epi32(2);
	__m256i one = _mm256_set1_epi32(1);
	__m256i zero = _mm256_set1_epi32(0);

	for (size i = 0; i < n / 8; ++i)
	{
		// Handle Collision between circles[cols[2*i]] and circles[cols[2*i+1]] circles
		const size start = static_cast<size>(i * 8);

		const __m256i offset = _mm256_set1_epi32(start);
		const __m256i addresses = _mm256_add_epi32(base, offset);
		const __m256i first_indexes = _mm256_i32gather_epi32(cols.data(), addresses, sizeof(size) * 2);
		const __m256i second_indexes = _mm256_i32gather_epi32(&cols[1], addresses, sizeof(size) * 2);

		const __m256 first_x = _mm256_i32gather_ps(
			reinterpret_cast<float*>(&circles.positions[0]),
			first_indexes,
			sizeof(float) * 2);
		const __m256 second_x = _mm256_i32gather_ps(
			reinterpret_cast<float*>(&circles.positions[0]),
			second_indexes,
			sizeof(float) * 2);
		const __m256 dx = _mm256_sub_ps(second_x, first_x);

		const __m256 first_y = _mm256_i32gather_ps(
			reinterpret_cast<float*>(&circles.positions[0].y),
			first_indexes,
			sizeof(float) * 2);
		const __m256 second_y = _mm256_i32gather_ps(
			reinterpret_cast<float*>(&circles.positions[0].y),
			second_indexes,
			sizeof(float) * 2);
		const __m256 dy = _mm256_sub_ps(second_y, first_y);

		const __m256 first_s = _mm256_i32gather_ps(
			reinterpret_cast<float*>(&circles.scales[0]),
			first_indexes,
			sizeof(float));
		const __m256 second_s = _mm256_i32gather_ps(
			reinterpret_cast<float*>(&circles.scales[0]),
			second_indexes,
			sizeof(float));

		const __m256 ds = _mm256_sub_ps(second_s, first_s);

		const __m256 dis = _mm256_sqrt_ps(_mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dx, dx)));

		const __m256 covered_by_2 = _mm256_div_ps(_mm256_sub_ps(ds, dis), _mm256_set1_ps(2));
	}
}