#pragma once

#include "util/linalg.h"
#include "util/random.h"
#include "util/simd.h"
#include <cmath>
#include <random>

namespace mesh {

/**
 * Special unitary group SU(3)
 *     - Can hold both group and algebra elements
 *     - T is expected to be one of float, double, simd<float>, simd<double>
 */
template <typename T> struct SU3
{
	using value_type = T;

	// machine-readable name of the group
	static constexpr std::string_view name() { return "su3"; }

	// dimension of the Lie group
	//   = number of generators
	//   = dimension of adjoint representation
	static constexpr int dim() { return 8; }

	// number of "colors"
	//   = dimension of defining/fundamental representation
	static constexpr int Nc() { return 3; }

	// constructors
	SU3() = default;
	explicit SU3(util::Matrix<util::complex<T>, 3> const &v) : v_(v) {}
	explicit SU3(T const &v) : v_(v) {}

	// special elements
	static SU3 zero() { return SU3(util::Matrix<util::complex<T>, 3>::zero()); }
	static SU3 one()
	{
		return SU3(util::Matrix<util::complex<T>, 3>::identity());
	}

	// uniform random group element
	template <typename Rng> static SU3 randomGroupElement(Rng &rng)
	{
		SU3 r;
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				r.v_(i, j) = {rng.template normal<T>(),
				              rng.template normal<T>()};
		r.v_ = gramSchmidt(r.v_);
		r.v_(0) /= determinant(r.v_);
		return r;
	}

	// algebra element with normal-random coefficnets and tr(T^aT^b) = 1/2 δ^ab
	template <typename Rng> static SU3 randomAlgebraElement(Rng &rng)
	{
		// TODO: this could be made faster (only 8 independent numbers, not 18)
		SU3 r;
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				r.v_(i, j) = {rng.template normal<T>(),
				              rng.template normal<T>()};
		return projectOnAlgebra(r) * sqrt(0.5);
	}

	util::Matrix<util::complex<T>, 3> v_;
	static constexpr size_t size() { return 18; }
	T *data() { return &v_.data()->re; }
	T const *data() const { return &v_.data()->re; }
};

///////////////////////////////////////////////////////////////////////////////
// ring structure of 3x3 matrices

template <typename T> SU3<T> operator-(SU3<T> const &a) { return SU3(-a.v_); }

template <typename T, typename U>
auto operator+(SU3<T> const &a, SU3<U> const &b)
{
	return SU3(a.v_ + b.v_);
}

template <typename T, typename U>
auto operator-(SU3<T> const &a, SU3<U> const &b)
{
	return SU3(a.v_ - b.v_);
}

template <typename T, typename U>
auto operator*(SU3<T> const &a, SU3<U> const &b)
{
	return SU3(a.v_ * b.v_);
}

template <typename T> void operator+=(SU3<T> &a, SU3<T> const &b)
{
	a.v_ += b.v_;
}

template <typename T> void operator-=(SU3<T> &a, SU3<T> const &b)
{
	a.v_ -= b.v_;
}

template <typename T> void operator*=(SU3<T> &a, SU3<T> const &b)
{
	a.v_ = a.v_ * b.v_;
}

template <typename T>
SU3<T> operator*(SU3<T> const &a, util::type_identity_t<T> b)
{
	return SU3(a.v_ * b);
}

template <typename T>
SU3<T> operator*(SU3<T> const &a, typename T::value_type b)
{
	return SU3(a.v_ * b);
}

template <typename T> void operator*=(SU3<T> &a, util::type_identity_t<T> b)
{
	a.v_ *= b;
}
template <typename T> void operator*=(SU3<T> &a, typename T::value_type b)
{
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			a.v_(i, j) *= b;
}

///////////////////////////////////////////////////////////////////////////////
// additional operations for the Lie group/algebra

// adjoint of group element
template <typename T> SU3<T> adj(SU3<T> const &a) { return SU3<T>(adj(a.v_)); }

// trace in defining (fundamental) representation
template <typename T> util::complex<T> trace(SU3<T> const &a)
{
	return trace(a.v_);
}

// squared frobenius of defining representation
template <typename T> T norm2(SU3<T> const &a) { return norm2(a.v_); }

// exponential map su(3) algebra -> SU(3) group
template <typename T> SU3<T> exp(SU3<T> const &a)
{
	// TODO: could be improved by assuming that a is indeed in the algebra
	return SU3<T>(exp(a.v_));
}

// projection to SU(3)
/*template <typename T> SU3<T> projectOnGroup(SU3<T> const &a)
{
    auto r = a * (adj(a) * a).inverse().sqrt();
    return r * std::pow(r.determinant(), -1.0 / 3);
}*/

// projection to SU(3) (only valid if already close)
template <typename T> SU3<T> projectOnGroupFast(SU3<T> const &a)
{
	auto r = a * 1.5 - a * adj(a) * a * 0.5;
	return SU3(r.v_ * (4.0 / 3.0 - (1 / 3.0) * determinant(r.v_)));
}

// projection to su(3) algebra (traceless anti-hermitian matrices)
template <typename T> SU3<T> projectOnAlgebra(SU3<T> const &a)
{
	auto b = (a - adj(a)) * 0.5;
	util::complex<T> tmp = trace(b) * (1.0 / 3.0);
	for (int i = 0; i < 3; ++i)
		b.v_(i, i) -= tmp;
	return b;
}

// simd operations

using util::vshuffle, util::vsum, util::vextract, util::vinsert;

template <typename T> auto vsum(SU3<T> const &a) { return SU3(vsum(a.v_)); }
template <typename T> auto vextract(SU3<T> const &a, size_t lane)
{
	return SU3(vextract(a.v_, lane));
}
template <typename T, typename U>
void vinsert(SU3<T> &a, size_t lane, SU3<U> const &b)
{
	vinsert(a.v_, lane, b.v_);
}

} // namespace mesh
