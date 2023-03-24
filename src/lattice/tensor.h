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

	static constexpr int simd_width = 1;
};

template <> struct TensorTraits<double>
{
	using Real = double;
	using vReal = double;
	using Object = double;
	using vObject = double;

	static constexpr int simd_width = 1;
};

template <typename T, int W> struct TensorTraits<util::simd<T, W>>
{
	using Real = T;
	using vReal = util::simd<T, W>;
	using Object = T;
	using vObject = util::simd<T, W>;
	static constexpr int simd_width = 1;
};

template <template <typename> typename F> struct TensorTraits<F<float>>
{
	using Real = float;
	using vReal = float;
	using Object = F<float>;
	using vObject = F<float>;

	static constexpr int simd_width = 1;
};

template <template <typename> typename F> struct TensorTraits<F<double>>
{
	using Real = double;
	using vReal = double;
	using Object = F<double>;
	using vObject = F<double>;

	static constexpr int simd_width = 1;
};

template <template <typename> typename F, typename T, int W>
struct TensorTraits<F<util::simd<T, W>>>
{
	using Real = T;
	using vReal = util::simd<T, W>;
	using Object = F<T>;
	using vObject = F<util::simd<T, W>>;

	static constexpr int simd_width = W;
};

} // namespace mesh
