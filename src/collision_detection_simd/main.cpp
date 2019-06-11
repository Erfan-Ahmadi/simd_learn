#include <vector>
#include <time.h>
#include "../common.hpp"

constexpr size_t instance_count = 1 << 3;
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
	srand(time(NULL));
	circles.resize(instance_count);
	cols.resize(cols_count * 2);

	for (size i = 0; i < instance_count; ++i)
	{
		circles.positions[i] =
		{
			static_cast<float>(rand() % 16),
			static_cast<float>(rand() % 16)
		};

		circles.velocities[i] =
		{
			i * 2.0f,
			i * 2.0f
		};

		circles.scales[i] = 5;
	}

	for (size_t i = 0; i < cols_count * 2; i += 2)
	{
		cols[i] = i % instance_count;
		cols[i + 1] = (i + 1) % instance_count;
	}
}

inline void handle_collision_simd()
{
	// With 256 bit Vector Instructions (8 floats), we should loop (cols_count / 8) times 
	const auto left = (cols_count % 8);
	const auto n = cols_count - left;

	__m256i base = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

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

int main()
{
	// Single Instruction Multiple Data
	std::cout << "Hello, SIMD Collision!" << std::endl;

	initialize_data();

	// SIMD
	const size_t twos = (instance_count * (instance_count - 1)) / 2;

	size_t row = 0;
	size_t column = 1;

	const size_t left = (twos % 8);
	const size_t n = twos - left;

	__m256i base = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	size* checks = static_cast<size*>(malloc(sizeof(size) * 8 * 2));
	size_t counter = 0;

	for (size_t k = 0; k < twos; ++k)
	{
		checks[counter] = row;
		checks[counter + 1] = column;
		
		counter += 2;

		if (counter >= 16)
		{
			counter = 0;

			__m256i first_indexes	=		_mm256_i32gather_epi32(&checks[0], base, sizeof(size) * 2);
			__m256i second_indexes	=		_mm256_i32gather_epi32(&checks[1], base, sizeof(size) * 2);

			print_vec_128<__m256i, int>(first_indexes);

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

			__m256 dis2 = _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dx, dx));

			const __m256 first_s = _mm256_i32gather_ps(
				reinterpret_cast<float*>(&circles.scales[0]),
				first_indexes,
				sizeof(float));
			const __m256 second_s = _mm256_i32gather_ps(
				reinterpret_cast<float*>(&circles.scales[0]),
				second_indexes,
				sizeof(float));

			const __m256 radii = _mm256_add_ps(first_s, second_s);
			__m256 radii2 = _mm256_mul_ps(radii, radii);
			__m256 result = _mm256_cmp_ps(dis2, radii2, _CMP_GT_OQ);

			float* resultf = reinterpret_cast<float*>(&result);
			for (short m = 0; m < 8; ++m)
			{
				if (resultf[m] == 0)
				{
					std::cout << "COLLIDED :" << checks[m * 2] << " and " << checks[m * 2 + 1] << std::endl;
				}
			}
		}
		else if (k >= twos - 1)
		{
			for (short m = 0; m < left; ++m)
			{
				const auto i = checks[m * 2];
				const auto j = checks[m * 2 + 1];
				const auto dx = circles.positions[i].x - circles.positions[j].x;
				const auto dy = circles.positions[i].y - circles.positions[j].y;
				const auto dis2 = (dy * dy + dx * dx);
				const auto radii = circles.scales[i] + circles.scales[j];

				if (dis2 <= radii * radii)
				{
					std::cout << "COLLIDED :" << checks[m * 2] << " and " << checks[m * 2 + 1] << std::endl;
				}
			}
		}

		if (++column > instance_count - 1)
		{
			row++;
			column = row + 1;
		}
	}

	free(checks);

	// Sequential
	for (size_t i = 0; i < instance_count; ++i)
	{
		for (size_t j = i + 1; j < instance_count; ++j)
		{
			const auto dx = circles.positions[i].x - circles.positions[j].x;
			const auto dy = circles.positions[i].y - circles.positions[j].y;
			const auto dis2 = (dy * dy + dx * dx);
			const auto radii = circles.scales[i] + circles.scales[j];

			if (dis2 <= radii * radii)
			{
			}
		}
	}
}