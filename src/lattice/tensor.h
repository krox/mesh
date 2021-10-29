#pragma once

/**
 * Type traits for handling the types inside a Lattice, which can be real or
 * complex, single or double precision, and have (potentially multiple) vector
 * or matrix indices. For example a Color-matrix can be stored as
 *     Matrix<complex<simd<double, W>>, Nc>
 * and a (pseudo-)fermion field as
 *     Vector<Vector<complex<simd<double>, W>, Nc>, Ns>
 */

#include "util/complex.h"
#include "util/linalg.h"
#include "util/simd.h"

namespace mesh {

template <typename T> struct TensorTraits;

template <> struct TensorTraits<float>
{
	using RealType = float;
	using ComplexType = util::complex<float>;
	using ScalarType = float;
	static constexpr size_t simdWidth = 1;
	using BaseType = float;
};

template <> struct TensorTraits<double>
{
	using RealType = double;
	using ComplexType = util::complex<double>;
	using ScalarType = double;
	static constexpr size_t simdWidth = 1;
	using BaseType = double;
};

template <typename T, size_t W> struct TensorTraits<util::simd<T, W>>
{
	using RealType = util::simd<typename TensorTraits<T>::RealType, W>;
	using ComplexType = util::simd<typename TensorTraits<T>::ComplexType, W>;
	using ScalarType = T;
	static constexpr size_t simdWidth = W;
	using BaseType = util::simd<T, W>;
};

template <typename T> struct TensorTraits<util::complex<T>>
{
	using RealType = T;
	using ComplexType = util::complex<T>;
	using ScalarType = util::complex<typename TensorTraits<T>::ScalarType>;
	static constexpr size_t simdWidth = TensorTraits<T>::simdWidth;
	using BaseType = typename TensorTraits<T>::BaseType;
};

template <typename T, size_t N> struct TensorTraits<util::Vector<T, N>>
{
	using RealType = util::Vector<typename TensorTraits<T>::RealType, N>;
	using ComplexType = util::Vector<typename TensorTraits<T>::ComplexType, N>;
	using ScalarType = util::Vector<typename TensorTraits<T>::ScalarType, N>;
	static constexpr size_t simdWidth = TensorTraits<T>::simdWidth;
	using BaseType = typename TensorTraits<T>::BaseType;
};

template <typename T, size_t N> struct TensorTraits<util::Matrix<T, N>>
{
	using RealType = util::Matrix<typename TensorTraits<T>::RealType, N>;
	using ComplexType = util::Matrix<typename TensorTraits<T>::ComplexType, N>;
	using ScalarType = util::Matrix<typename TensorTraits<T>::ScalarType, N>;
	static constexpr size_t simdWidth = TensorTraits<T>::simdWidth;
	using BaseType = typename TensorTraits<T>::BaseType;
};

} // namespace mesh
