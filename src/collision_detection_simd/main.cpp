#include <iostream>
#include <immintrin.h>
#include <vector>

constexpr int instance_count = 1 << 4;
constexpr int cols_count = 16;
static_assert(cols_count <= instance_count * 2, "collision count larger than it should be.");
typedef size_t size;

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
			static_cast<float>(i)
		};

		circles.velocities[i] =
		{
			i * 2.0f,
			i * 2.0f
		};

		circles.scales[i] = 10;
	}

	for (size i = 0; i < cols_count; i += 2)
	{
		cols[i] = i % instance_count;
		cols[i + 1] = (i + 1) % instance_count;
	}
}

inline void print_m256_float(__m256& vec)
{
	float* vecf = reinterpret_cast<float*>(&vec);

	for (short i = 0; i < 8; ++i)
		std::cout << "(" << i << "): " << vecf[i] << std::endl;
}

int main()
{
	// Single Instruction Multiple Data
	std::cout << "Hello, SIMD Collision!" << std::endl;

	initialize_data();

	// We assume collisions are recorded in cols vector, no collision detection is done!

	// With 256 bit Vector Instructions (8 floats), we should loop (cols_count / 8) times 

	for (size i = 0; i < cols_count / 8; ++i)
	{
		// Handle Collision between circles[cols[2*i]] and circles[cols[2*i+1]] circles
		const uint32_t start = static_cast<uint32_t>(i * 16);

		// Get Pos X's
		__m256i index = _mm256_setr_epi32(
			start,
			start + 2,
			start + 4,
			start + 6,
			start + 8,
			start + 10,
			start + 12,
			start + 14);
		__m256 posx = _mm256_i32gather_ps(
			reinterpret_cast<float*>(circles.positions.data()),
			index,
			4);
		
		// Get Pos Y's
		index = _mm256_setr_epi32(
			start + 1,
			start + 3,
			start + 5,
			start + 7,
			start + 9,
			start + 11,
			start + 13,
			start + 15);
		__m256 posy = _mm256_i32gather_ps(
			reinterpret_cast<float*>(circles.positions.data()),
			index,
			4);

		print_m256_float(posx);
		print_m256_float(posy);
	}
}