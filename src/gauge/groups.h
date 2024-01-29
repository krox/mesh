#pragma once

#include "util/complex.h"
#include "util/linalg.h"
#include <type_traits>

// Definitions of and utilities for the gauge groups.
// * This module deals with individual elements of the group/algebra, operations
//   on whole lattice objects are located in gauge/utils.h for example.
// * Ideally, this would be the only file that contains code dependent on
//   specific gauge groups. Everything else should be templated over arbitrary
//   groups. Probably not quite true in practice of course.

namespace mesh {

namespace gauge {

// definitions of the gauge groups
// NOTE: In a previous version, there were actual types for the group/algebra
//       elements. But for example SU3<T> was just a very shallow wrapper around
//       Matrix<complex<T>,3> which resulted in a lot of boilerplate. Now we use
//       the internal types directly and just effectively assume that
//       'Matrix<complex>' is meant to represent an element of SU(3) (or su(3))
//       depending on context.
// TODO: For better efficiency, we should have at least three types per gauge
//       group: One for the Lie algebra, one for the Lie group, and (at least)
//       one for an actual representation.

////////////////////////////////////////////////////////////////////////////////
// U(1)
////////////////////////////////////////////////////////////////////////////////

template <class T> using U1 = util::complex<T>;

template <class G> struct GaugeTraits;
template <std::floating_point T> struct GaugeTraits<U1<T>>
{
	// machine-readable name of the group
	static constexpr std::string_view name() { return "u1"; }

	// (real) dimension of the Lie group / algebra
	//   = number of generators
	//   = dimension of adjoint representation
	static constexpr int dim() { return 1; }

	// number of "colors"
	//   = dimension of defining/fundamental representation
	//   = trace of unity
	static constexpr int Nc() { return 1; }

	// number of 'T' multiplications per 'group_type' multiplications
	static constexpr int mul_complexity() { return 4; }
};

template <class T> UTIL_DEVICE U1<T> adj(U1<T> const &a) { return conj(a); }

// Real part of trace in the "defining"/"fundamental"/"vector" representation.
// This is used in most gauge actions.
template <class T> UTIL_DEVICE T real_trace(U1<T> const &a) { return a.re; }

// project group to algebra.
// This is a linear approximation of 'log' around unity.
// (used in Landau gauge fixing for example)
template <class T> UTIL_DEVICE U1<T> project_on_algebra(U1<T> const &a)
{
	return {T(0), a.im};
}

// project a to the group manifold
// (used in some smearing procedures for example)
template <class T> UTIL_DEVICE U1<T> project_on_group(U1<T> const &a)
{
	using std::sqrt;
	return a * (1.0 / sqrt(norm2(a.v_)));
}

// Fast approximation of project_on_group, valid if a is already close to
// the manifold. Used to correct rounding errors during HMC.
template <class T> UTIL_DEVICE U1<T> project_on_group_fast(U1<T> const &a)
{
	// just a linear approximation of the 'sqrt' from the exact formula
	return a * (1.5 - norm2(a) * 0.5);
}

// algebra element with normal-random coeffs and tr(T^aT^b) = 1/2 δ^ab
template <class T, class Rng> void random_algebra_element(U1<T> &a, Rng &rng)
{
	// the sole generator of U(1) is iT = "1/sqrt(2) i"
	a.re = T(0);
	a.im = rng.template normal<T>() * T(M_SQRT1_2);
}

// uniform-random group element
template <class T, class Rng> void random_group_element(U1<T> &a, Rng &rng)
{
	using std::sin, std::cos;
	T t = rng.template uniform<T>() * (2 * M_PI);
	a.re = cos(t);
	a.im = sin(t);
}

// exponential map algebra -> group
template <class T> UTIL_DEVICE U1<T> exp(U1<T> const &a)
{
	// we assume that a is indeed in the algebra, i.e. a.real == 0
	using std::cos, std::sin;
	return {cos(a.im), sin(a.im)};
}

////////////////////////////////////////////////////////////////////////////////
// SU(2)
////////////////////////////////////////////////////////////////////////////////

template <class T> using SU2 = util::quaternion<T>;

template <std::floating_point T> struct GaugeTraits<SU2<T>>
{
	// NOTE: this group is regularly written in two equivalent ways:
	// 1) as complex 2x2 matrices of the form
	//        v0 + i*v_k*sigma_k  = |  v0 + i v3   v2 + i v1  |
	//                              | -v2 + i v1   v0 - i v3  |
	//    with v0^2 + v1^2 + v2^2 + v3^2 == 1
	// 2) as normalized quaternions v0 + v1*I + v2*J + v3*K
	//
	// we use `util::quaternion` because it implements all the arithmetic we
	// need, even though in physics we usually imagine 2x2 matrices which gives
	// direct meaning to linear algebra terms like 'trace' and 'determinant'

	static constexpr std::string_view name() { return "su2"; }
	static constexpr int dim() { return 3; }
	static constexpr int Nc() { return 2; }
	static constexpr int mul_complexity() { return 16; }
};

template <class T> UTIL_DEVICE SU2<T> adj(SU2<T> const &a) { return conj(a); }

template <class T> UTIL_DEVICE T real_trace(SU2<T> const &a)
{
	return a.re * 2.0;
}

template <class T> UTIL_DEVICE SU2<T> project_on_algebra(SU2<T> const &a)
{
	return {T(0), a.im1, a.im2, a.im3};
}

template <class T> UTIL_DEVICE SU2<T> project_on_group(SU2<T> const &a)
{
	using std::sqrt;
	return a * (1.0 / sqrt(norm2(a.v_)));
}

template <class T> UTIL_DEVICE SU2<T> exp(SU2<T> const &a)
{
	// Assuming a.re==0, the following is exact, and also faster than a
	// typical 12-order Taylor series. Sadly, this version produces
	//     exp({0,0,0,0}) = {1, nan, nan, nan}
	// though hopefully that does not happen in practice
	// TODO: replace sin(alpha)/alpha by a proper implementation of 'sinc'
	using std::sqrt, std::sin, std::cos;
	T alpha = sqrt(a.im1 * a.im1 + a.im2 * a.im2 + a.im3 * a.im3);
	T f = sin(alpha) / alpha;
	return {cos(alpha), a.im1 * f, a.im2 * f, a.im3 * f};
}

template <class T, class Rng> void random_algebra_element(SU2<T> &a, Rng &rng)
{
	// NOTE: The properly normalized generators are
	//       iT^1 = (0, 1/2, 0, 0)
	//       iT^2 = (0, 0, 1/2, 0)
	//       iT^3 = (0, 0, 0, 1/2)
	a.re = T(0);
	a.im1 = 0.5 * rng.template normal<T>();
	a.im2 = 0.5 * rng.template normal<T>();
	a.im3 = 0.5 * rng.template normal<T>();
}

template <class T, class Rng> void random_group_element(SU2<T> &a, Rng &rng)
{
	using std::sqrt;
	a.re = rng.template normal<T>();
	a.im1 = rng.template normal<T>();
	a.im2 = rng.template normal<T>();
	a.im3 = rng.template normal<T>();
	a *= 1.0 / sqrt(norm2(a));
}

template <class T> UTIL_DEVICE SU2<T> project_on_group_fast(SU2<T> const &a)
{
	// just a linear approximation of the 'sqrt' from the exact formula
	return a * (1.5 - norm2(a) * 0.5);
}

////////////////////////////////////////////////////////////////////////////////
// SU(N), N >= 3
////////////////////////////////////////////////////////////////////////////////

template <class T, int N>
    requires(N >= 3)
using SU = util::Matrix<util::complex<T>, N>;
template <class T> using SU3 = SU<T, 3>;
template <class T> using SU4 = SU<T, 4>;
template <class T> using SU5 = SU<T, 5>;
template <class T> using SU6 = SU<T, 6>;

template <std::floating_point T, int N> struct GaugeTraits<SU<T, N>>
{
	static constexpr std::string name() { return fmt::format("su{}", N); }
	static constexpr int dim() { return N * N - 1; }
	static constexpr int Nc() { return N; }
	static constexpr int mul_complexity() { return N * N * N * 4; }
};

template <class T, int N> UTIL_DEVICE SU<T, N> adj(SU<T, N> const &a)
{
	return util::adj(a);
}

template <class T, int N> UTIL_DEVICE T real_trace(SU<T, N> const &a)
{
	return util::trace(a).re;
}

template <class T, int N>
UTIL_DEVICE SU<T, N> project_on_algebra(SU<T, N> const &a)
{
	return antihermitian_traceless(a);
}

template <class T, int N>
UTIL_DEVICE SU<T, N> project_on_group(SU<T, N> const &a)
{
	// the exact formula contains a matrix-inverse-square-root...
	assert(false);
}

template <class T, int N>
UTIL_DEVICE SU<T, N> project_on_group_fast(SU<T, N> const &a)
{
	assert(N == 3); // havnt checked if the formula below is specific to N=3
	auto r = a * 1.5 - a * adj(a) * a * 0.5;
	return r * (4.0 / 3.0 - (1 / 3.0) * util::determinant(r));
}

template <class T, int N, class Rng>
void random_algebra_element(SU<T, N> &a, Rng &rng)
{
	// TODO: this could be made faster (only approx N^2/2 independent
	// numbers, not N^2)
	a = util::Matrix<util::complex<T>, N>::random_normal(rng);
	a = project_on_algebra(a);
	a *= sqrt(0.5);
}

template <class T, int N, class Rng>
void random_group_element(SU<T, N> &a, Rng &rng)
{
	a = util::Matrix<util::complex<T>, N>::random_normal(rng);
	a = gram_schmidt(a);
	a(0) /= determinant(a);
}

template <class T, int N> UTIL_DEVICE SU<T, N> exp(SU<T, N> const &a)
{
	// TODO: this could be a lot faster using the fact that a is
	// antihermitian-traceless
	return util::exp(a);
}

} // namespace gauge

// despite this 'using' directive, the 'gauge' namespace can still be useful to
// solve some ambiguities. For example 'gauge::exp(...)' always refers to to the
// algebra -> group mapping, wheras 'exp' alone might be a general matrix
// exponential (theoretically the same, but potentially slower)
using namespace gauge;

} // namespace mesh