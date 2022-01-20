#pragma once

/**
 * Type traits for handling the types inside a Lattice, which can be real or
 * complex, single or double precision, and have (potentially multiple) vector
 * or matrix indices. For example a Color-matrix can be stored as
 *     Matrix<complex<simd<double, W>>, Nc>
 * and a (pseudo-)fermion field as
 *     Vector<Vector<complex<simd<double>, W>, Nc>, Ns>
 *
 * TODO: Actually, I think some of this can be removed:
 *       - real/complex conversion is not necessary if strong types are used,
 *         I.e. SU3<T> instead of Matrix<complex<T>,3>. The gauge group itself
 *         knows what should or should not be complex.
 *       - scalar/simd conversion should not be done this way. Use
 *             operator*(SU3<simd<T>>, T)
 *         instead of
 *             operator*(SU3<T>, remove_simd_t<T>)
 *         because the latter one potentially clashes with the primary overload.
 *         An even better signature might be
 *             operator*(SU3<simd<T>>, type_identity_t<T>)
 *         because it allows implicit conversions of the scalar value without
 *         producing conflicting template parameter deductions.
 *       - the 'BaseType' and 'simdWidth' are only used in some Lattice<T>
 *         operatorions (e.g. cshift) that assume all tensor types to be
 *         structured as a simple array of T. There should be a cleaner way
 *         to do this.
 *       - float/double conversion I have to think about some more...
 */

#include "util/complex.h"
#include "util/linalg.h"
#include "util/simd.h"

namespace mesh {

template <typename T> struct TensorTraits;

template <> struct TensorTraits<float>
{
	using ScalarType = float;
	static constexpr size_t simdWidth = 1;
	using BaseType = float;
};

template <> struct TensorTraits<double>
{
	using ScalarType = double;
	static constexpr size_t simdWidth = 1;
	using BaseType = double;
};

template <typename T, size_t W> struct TensorTraits<util::simd<T, W>>
{
	using ScalarType = T;
	static constexpr size_t simdWidth = W;
	using BaseType = util::simd<T, W>;
};

template <typename T> struct TensorTraits<util::complex<T>>
{
	using ScalarType = util::complex<typename TensorTraits<T>::ScalarType>;
	static constexpr size_t simdWidth = TensorTraits<T>::simdWidth;
	using BaseType = typename TensorTraits<T>::BaseType;
};

template <typename T, size_t N> struct TensorTraits<util::Vector<T, N>>
{
	using ScalarType = util::Vector<typename TensorTraits<T>::ScalarType, N>;
	static constexpr size_t simdWidth = TensorTraits<T>::simdWidth;
	using BaseType = typename TensorTraits<T>::BaseType;
};

template <typename T, size_t N> struct TensorTraits<util::Matrix<T, N>>
{
	using ScalarType = util::Matrix<typename TensorTraits<T>::ScalarType, N>;
	static constexpr size_t simdWidth = TensorTraits<T>::simdWidth;
	using BaseType = typename TensorTraits<T>::BaseType;
};

} // namespace mesh
