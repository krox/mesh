#pragma once

/**
 * Helper structure, with type traits for Lattice
 */

#include "util/simd.h"

namespace mesh {

template <typename> struct TensorTraits;

template <> struct TensorTraits<float>
{
	using Real = float;
	using vReal = float;
	using Object = float;
	using vObject = float;

	static constexpr size_t simd_width = 1;
};

template <> struct TensorTraits<double>
{
	using Real = double;
	using vReal = double;
	using Object = double;
	using vObject = double;

	static constexpr size_t simd_width = 1;
};

template <typename T, size_t W> struct TensorTraits<util::simd<T, W>>
{
	using Real = T;
	using vReal = util::simd<T, W>;
	using Object = T;
	using vObject = util::simd<T, W>;
};

template <template <typename> typename F, typename T> struct TensorTraits<F<T>>
{
	using Real = T;
	using vReal = T;
	using Object = F<T>;
	using vObject = F<T>;

	static constexpr size_t simd_width = 1;
};

template <template <typename> typename F, typename T, size_t W>
struct TensorTraits<F<util::simd<T, W>>>
{
	using Real = T;
	using vReal = util::simd<T, W>;
	using Object = F<T>;
	using vObject = F<util::simd<T, W>>;

	static constexpr size_t simd_width = W;
};

} // namespace mesh
