#pragma once

#include <iostream>
#include <immintrin.h>

template <class V, class T>
inline void print_vec(V& vec)
{
	T* vecf = reinterpret_cast<T*>(&vec);

	for (short i = 0; i < 8; ++i)
		std::cout << "(" << i << "): " << vecf[i] << std::endl;

	std::cout << std::endl;
}