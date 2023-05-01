#pragma once

#include "fmt/format.h"
#include "util/linalg.h"
#include "util/random.h"
#include <cmath>
#include <random>

namespace mesh {

// TODO: SU<T,N> is a thin wrapper around util::Matrix<util::complex<T>, N>. We
//       could probably get rid of it and use Matrix directly. Possibly with a
//       traits-type to get the group structure right. Same for
//           U1 -> util::complex
//           SU2 -> util::quaternion (which doesnt exist yet)

// utils linalg.h does not implement determinant above 3x3 matrices.
// we define it here to get some benchmarks compiling.
// TODO: actually implement it
template <typename T, int N>
    requires(N >= 4)
T determinant(util::Matrix<T, N> const &)
{
	return T(1.0);
}

/**
 * Special unitary group SU(N)
 *     - Can hold both group and algebra elements
 *     - T is expected to be one of float, double, simd<float>, simd<double>
 */
template <typename T, int N> struct SUN
{
	static_assert(N >= 3);

	using value_type = T;

	// machine-readable name of the group
	static constexpr std::string name() { return fmt::format("su{}", N); }

	// dimension of the Lie group
	//   = number of generators
	//   = dimension of adjoint representation
	static constexpr int dim() { return N * N - 1; }

	// number of "colors"
	//   = dimension of defining/fundamental representation
	static constexpr int Nc() { return N; }

	// number of T multiplications for SUN<T,N> multiplications
	static constexpr int mul_complexity() { return N * N * N * 4; }

	// constructors
	SUN() = default;
	explicit SUN(util::Matrix<util::complex<T>, N> const &v) : v_(v) {}
	explicit SUN(T const &v) : v_(v) {}

	// special elements
	static SUN zero() { return SUN(util::Matrix<util::complex<T>, N>::zero()); }
	static SUN one()
	{
		return SUN(util::Matrix<util::complex<T>, N>::identity());
	}

	// uniform random group element
	template <typename Rng> static SUN randomGroupElement(Rng &rng)
	{
		SUN r;
		r.v_ = util::Matrix<util::complex<T>, N>::random_normal(rng);
		r.v_ = gram_schmidt(r.v_);
		r.v_(0) /= determinant(r.v_);
		return r;
	}

	// algebra element with normal-random coefficnets and tr(T^aT^b) = 1/2 δ^ab
	template <typename Rng> static SUN randomAlgebraElement(Rng &rng)
	{
		// TODO: this could be made faster (only approx N^2/2 independent
		// numbers, not N^2)
		SUN r;
		r.v_ = util::Matrix<util::complex<T>, N>::random_normal(rng);
		return projectOnAlgebra(r) * sqrt(0.5);
	}

	util::Matrix<util::complex<T>, N> v_;
	static constexpr size_t size() { return 2 * N * N; }
	T *data() { return &v_.data()->re; }
	T const *data() const { return &v_.data()->re; }
};

///////////////////////////////////////////////////////////////////////////////
// ring structure of NxN matrices

template <typename T, int N> SUN<T, N> operator-(SUN<T, N> const &a)
{
	return SUN(-a.v_);
}

template <typename T, typename U, int N>
auto operator+(SUN<T, N> const &a, SUN<U, N> const &b)
{
	return SUN(a.v_ + b.v_);
}

template <typename T, typename U, int N>
auto operator-(SUN<T, N> const &a, SUN<U, N> const &b)
{
	return SUN(a.v_ - b.v_);
}

template <typename T, typename U, int N>
auto operator*(SUN<T, N> const &a, SUN<U, N> const &b)
{
	return SUN(a.v_ * b.v_);
}

template <typename T, int N> void operator+=(SUN<T, N> &a, SUN<T, N> const &b)
{
	a.v_ += b.v_;
}

template <typename T, int N> void operator-=(SUN<T, N> &a, SUN<T, N> const &b)
{
	a.v_ -= b.v_;
}

template <typename T, int N> void operator*=(SUN<T, N> &a, SUN<T, N> const &b)
{
	a.v_ = a.v_ * b.v_;
}

template <typename T, int N>
SUN<T, N> operator*(SUN<T, N> const &a, std::type_identity_t<T> b)
{
	return SUN(a.v_ * b);
}

template <typename T, int N>
SUN<T, N> operator*(SUN<T, N> const &a, typename T::value_type b)
{
	return SUN(a.v_ * b);
}

template <typename T, int N>
void operator*=(SUN<T, N> &a, std::type_identity_t<T> b)
{
	a.v_ *= b;
}
template <typename T, int N>
void operator*=(SUN<T, N> &a, typename T::value_type b)
{
	for (int i = 0; i < N; ++i)
		for (int j = 0; j < N; ++j)
			a.v_(i, j) *= b;
}

///////////////////////////////////////////////////////////////////////////////
// additional operations for the Lie group/algebra

// adjoint of group element
template <typename T, int N> SUN<T, N> adj(SUN<T, N> const &a)
{
	return SUN<T, N>(adj(a.v_));
}

// trace in defining (fundamental) representation
template <typename T, int N> util::complex<T> trace(SUN<T, N> const &a)
{
	return trace(a.v_);
}

// squared frobenius of defining representation
template <typename T, int N> T norm2(SUN<T, N> const &a) { return norm2(a.v_); }

// exponential map su(3) algebra -> SU(3) group
template <typename T, int N> SUN<T, N> exp(SUN<T, N> const &a)
{
	// TODO: could be improved by assuming that a is indeed in the algebra
	return SUN<T, N>(exp(a.v_));
}

// projection to SU(3)
/*template <typename T> SUN<T,N> projectOnGroup(SUN<T,N> const &a)
{
    auto r = a * (adj(a) * a).inverse().sqrt();
    return r * std::pow(r.determinant(), -1.0 / 3);
}*/

// projection to SU(3) (only valid if already close)
template <typename T> SUN<T, 3> projectOnGroupFast(SUN<T, 3> const &a)
{
	auto r = a * 1.5 - a * adj(a) * a * 0.5;
	return SUN(r.v_ * (4.0 / 3.0 - (1 / 3.0) * util::determinant(r.v_)));
}

// projection to su(N) algebra (traceless anti-hermitian matrices)
template <typename T, int N> SUN<T, N> projectOnAlgebra(SUN<T, N> const &a)
{
	return SUN(antihermitian_traceless(a.v_));
}

template <class T> using SU3 = SUN<T, 3>;
template <class T> using SU4 = SUN<T, 4>;
template <class T> using SU5 = SUN<T, 5>;
template <class T> using SU6 = SUN<T, 6>;

} // namespace mesh
