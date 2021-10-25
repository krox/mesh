#pragma once

#include <cmath>
#include <random>

#include "util/linalg.h"
#include "util/random.h"

namespace mesh {

/**
 * Special unitary group SU(3)
 *     - Can hold both group and algebra elements
 *     - T is expected to be one of float, double, simd<float>, simd<double>
 */
template <typename T> struct SU3
{
	util::Matrix<util::complex<T>, 3> v_;

	// constructors
	SU3() = default;
	explicit SU3(util::Matrix<util::complex<T>, 3> const &v) : v_(v) {}
	explicit SU3(T const &v) : v_(v) {}
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
				r.v_(i, j) = {rng.normal(), rng.normal()};
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
				r.v_(i, j) = {rng.normal(), rng.normal()};
		return projectOnAlgebra(r) * sqrt(0.5);
	}
};

// unary SU3

// trace in defining (fundamental) representation
template <typename T> util::complex<T> trace(SU3<T> const &a)
{
	return trace(a.v_);
}

// adjoint of group element
template <typename T> SU3<T> adj(SU3<T> const &a) { return SU3<T>(adj(a.v_)); }

// projection to su(3) algebra (traceless anti-hermitian matrices)
template <typename T> SU3<T> projectOnAlgebra(SU3<T> const &a)
{
	auto b = (a - adj(a)) * 0.5;
	util::complex<T> tmp = trace(b) * T(1.0 / 3.0);
	for (int i = 0; i < 3; ++i)
		b.v_(i, i) -= tmp;
	return b;
}

// TODO: remove this, not really cleanly defined
template <typename T> T norm2(SU3<T> const &a) { return norm2(a.v_); }

// projection to SU(3) (only valid if already close)
template <typename T> SU3<T> projectOnGroup(SU3<T> const &a)
{
	// exact version
	// auto r = a * (adj(a) * a).inverse().sqrt();
	// return r * std::pow(r.determinant(), -1.0 / 3);

	// approximate version (if a is already close to SU(3))
	auto r = a * 1.5 - a * adj(a) * a * 0.5;
	return r * (4.0 / 3.0 - (1 / 3.0) * determinant(r.v_));
}

// exponential map su(3) algebra -> SU(3) group
template <typename T> SU3<T> exp(SU3<T> const &a)
{
	// TODO: could be improved by assuming that a is indeed in the algebra
	return SU3<T>(exp(a.v_));
}

// binary SU3 <-> SU3

template <typename T, typename U>
auto operator*(SU3<T> const &a, SU3<U> const &b)
{
	return SU3(a.v_ * b.v_);
}

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

// binary SU3 <-> scalar

template <typename T>
SU3<T> operator*(SU3<T> const &a, util::complex<T> const &b)
{
	return SU3(a.v_ * b);
}

template <typename T> SU3<T> operator*(SU3<T> const &a, double b)
{
	return SU3(a.v_ * b);
}

// op-assign
template <typename T> SU3<T> &operator+=(SU3<T> &a, SU3<T> const &b)
{
	a.v_ += b.v_;
	return a;
}
template <typename T> SU3<T> &operator-=(SU3<T> &a, SU3<T> const &b)
{
	a.v_ -= b.v_;
	return a;
}
template <typename T> SU3<T> &operator*=(SU3<T> &a, SU3<T> const &b)
{
	a.v_ *= b.v_;
	return a;
}

// simd operations

template <typename T> auto vsum(SU3<T> const &a) { return SU3(vsum(a.v_)); }
template <typename T, typename U> auto vshuffle(SU3<T> const &a, U const &mask)
{
	return SU3(vshuffle(a.v_, mask));
}
template <typename T> auto vextract(SU3<T> const &a, size_t lane)
{
	return SU3(vextract(a.v_, lane));
}
template <typename T, typename U>
void vinsert(SU3<T> &a, size_t lane, SU3<U> const &b)
{
	vinsert(a.v_, lane, b.v_);
}

template <typename> struct TensorTraits;
template <typename T> struct TensorTraits<SU3<T>>
{
	using ScalarType = SU3<typename TensorTraits<T>::ScalarType>;
	static constexpr size_t simdWidth = TensorTraits<T>::simdWidth;
};

} // namespace mesh
