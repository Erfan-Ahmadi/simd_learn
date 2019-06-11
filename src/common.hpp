#pragma once

#include <iostream>
#include <immintrin.h>

template <class V, class T>
inline void print_vec_256(V& vec)
{
	T* vecf = reinterpret_cast<T*>(&vec);
	
	std::cout << "(";
	for (short i = 0; i < 8; ++i)
		std::cout <<  vecf[i] << ((i == 7) ? ")" : ", ");

	std::cout << std::endl;
}

template <class V, class T>
inline void print_vec_128(V& vec)
{
	T* vecf = reinterpret_cast<T*>(&vec);
	
	std::cout << "(";
	for (short i = 0; i < 4; ++i)
		std::cout <<  vecf[i] << ((i == 3) ? ")" : ", ");

	std::cout << std::endl;
}