#pragma once

/**
 * Helper structure, with type traits for Lattice
 */

#include "util/simd.h"
#include <type_traits>

namespace mesh {

template <class> struct TensorTraits;

template <class R>
    requires std::is_floating_point_v<R>
struct TensorTraits<R>
{
	using Real = R;
	using vReal = R;
	using Object = R;
	using vObject = R;

	static constexpr int simd_width = 1;
	static constexpr int flops_mul = 1;
};

template <class R, int W>
    requires std::is_floating_point_v<R>
struct TensorTraits<util::simd<R, W>>
{
	using Real = R;
	using vReal = util::simd<R, W>;
	using Object = R;
	using vObject = util::simd<R, W>;
	static constexpr int simd_width = W;
	static constexpr int flops_mul = W;
};

template <template <class> class F, class R>
    requires std::is_floating_point_v<R>
struct TensorTraits<F<R>>
{
	using Real = R;
	using vReal = R;
	using Object = F<R>;
	using vObject = F<R>;

	static constexpr int simd_width = 1;
	static constexpr int flops_mul = F<R>::mul_complexity();
};

template <template <class, int> class F, class R, int N>
    requires std::is_floating_point_v<R>
struct TensorTraits<F<R, N>>
{
	using Real = R;
	using vReal = R;
	using Object = F<R, N>;
	using vObject = F<R, N>;

	static constexpr int simd_width = 1;
	static constexpr int flops_mul = F<R, N>::mul_complexity();
};

template <template <class> class F, class R, int W>
    requires std::is_floating_point_v<R>
struct TensorTraits<F<util::simd<R, W>>>
{
	using Real = R;
	using vReal = util::simd<R, W>;
	using Object = F<R>;
	using vObject = F<util::simd<R, W>>;

	static constexpr int simd_width = W;
	static constexpr int flops_mul = F<R>::mul_complexity() * W;
};

template <template <class, int> class F, int N, class R, int W>
    requires std::is_floating_point_v<R>
struct TensorTraits<F<util::simd<R, W>, N>>
{
	using Real = R;
	using vReal = util::simd<R, W>;
	using Object = F<R, N>;
	using vObject = F<util::simd<R, W>, N>;

	static constexpr int simd_width = W;
	static constexpr int flops_mul = F<R, N>::mul_complexity() * W;
};

} // namespace mesh
