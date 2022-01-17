#pragma once

#include "util/complex.h"
#include "util/random.h"
#include <cassert>
#include <cmath>
#include <random>

namespace mesh {

// unitary group U(1)
template <typename T> struct U1
{
	util::complex<T> v_;

	static constexpr int Nc() { return 1; }

	// constructors
	U1() = default;
	explicit U1(util::complex<T> const &v) : v_(v) {}
	static U1 zero() { return U1(T(0)); }
	static U1 one() { return U1(T(1)); }

	template <typename Rng> static U1<T> randomGroupElement(Rng &rng)
	{
		auto t = rng.uniform() * (2 * M_PI);
		return U1({std::cos(t), std::sin(t)});
	}

	// algebra element with normal-random coeffs and tr(T^aT^b) = 1/2 δ^ab
	template <typename Rng> static U1<T> randomAlgebraElement(Rng &rng)
	{
		// the sole generator of U(1) is iT = "1/sqrt(2) i"
		return U1<T>({T(0), rng.normal() * M_SQRT1_2});
	}
};

// TODO: either remove this, or define it more clearly
template <typename T> T norm2(U1<T> const &a) { return norm2(a.v_); }

template <typename T> util::complex<T> trace(U1<T> const &a) { return a.v_; }
template <typename T> U1<T> adj(U1<T> const &a) { return U1(conj(a.v_)); }
template <typename T> U1<T> projectOnAlgebra(U1<T> const &a)
{
	return U1<T>({T(0), a.v_.im});
}

template <typename T> U1<T> projectOnGroup(U1<T> const &a)
{
	return U1<T>(a.v_ / sqrt(norm2(a.v_)));
}

// scalar operator
template <typename T> U1<T> operator*(U1<T> const &a, double b)
{
	return U1<T>(a.v_ * b);
}
template <typename T> U1<T> operator/(U1<T> const &a, double b)
{
	return U1<T>(a.b_ / b);
}
template <typename T> void operator*=(U1<T> &a, double b) { a.v_ *= b; }
template <typename T> void operator/=(U1<T> &a, double b) { a.v_ /= b; }

// group/algebra operations
template <typename T> U1<T> operator+(U1<T> const &a, U1<T> const &b)
{
	return U1<T>(a.v_ + b.v_);
}
template <typename T> U1<T> operator-(U1<T> const &a, U1<T> const &b)
{
	return U1<T>(a.v_ - b.v_);
}
template <typename T> U1<T> operator*(U1<T> const &a, U1<T> const &b)
{
	return U1<T>(a.v_ * b.v_);
}

template <typename T> void operator+=(U1<T> &a, U1<T> const &b)
{
	a.v_ += b.v_;
}
template <typename T> void operator-=(U1<T> &a, U1<T> const &b)
{
	a.v_ -= b.v_;
}
template <typename T> void operator*=(U1<T> &a, U1<T> const &b)
{
	a.v_ *= b.v_;
}

// exponential map u(1) algebra -> U(1) group
template <typename T> U1<T> exp(U1<T> const &a)
{
	// we assume that a is indeed in the algebra, i.e. a.v_.real = 0
	return U1<T>({cos(a.v_.im), sin(a.v_.im)});
}

// simd operations

template <typename T> auto vsum(U1<T> const &a) { return U1(vsum(a.v_)); }
template <typename T> auto vextract(U1<T> const &a, size_t lane)
{
	return U1(vextract(a.v_, lane));
}
template <typename T, typename U>
void vinsert(U1<T> &a, size_t lane, U1<U> const &b)
{
	vinsert(a.v_, lane, b.v_);
}

template <typename> struct TensorTraits;
template <typename T> struct TensorTraits<U1<T>>
{
	using ScalarType = U1<typename TensorTraits<T>::ScalarType>;
	static constexpr size_t simdWidth = TensorTraits<T>::simdWidth;
	using BaseType = typename TensorTraits<T>::BaseType;
};

#if 0 // old stuff for heatbath algorithms

/** (unnormalized) probability distribution */
static double dist(double alpha, double alpha2, double tr)
{
	return exp(alpha * tr + alpha2 * tr * tr) * pow(1 - tr * tr, -0.5);
}

inline U1 U1::random(rng_t &rng)
{
	double phi = std::uniform_real_distribution<double>(-M_PI, M_PI)(rng);
	return U1(cos(phi), sin(phi));
}

inline U1 U1::random(rng_t &rng, double alpha)
{
	assert(alpha >= 0);

	// moderate alpha -> rejection algorithm based on uniform distribution
	if (alpha < 1)
	{
		// approximate distribution of phi (for small alpha)
		std::uniform_real_distribution<double> uni_dist(-M_PI, M_PI);

		while (true)
		{
			++nTries;
			double phi = uni_dist(rng);
			double r = cos(phi) - 1;
			// assert(r <= 0);
			if (!std::bernoulli_distribution(exp(alpha * r))(rng))
				continue;
			++nAccepts;
			return U1(cos(phi), sin(phi));
		}
	}

	// large alpha -> rejection algorithm based on normal distribution
	else
	{
		// approximate distribution of phi (for larger alpha)
		std::normal_distribution<double> norm_dist(0, 0.5 * M_PI / sqrt(alpha));

		while (true)
		{
			++nTries;
			auto phi = norm_dist(rng);
			if (phi > M_PI || phi < -M_PI)
				continue;
			double r = cos(phi) + (2 / M_PI / M_PI) * phi * phi - 1;
			// assert(r <= 0);
			if (!std::bernoulli_distribution(exp(alpha * r))(rng))
				continue;
			++nAccepts;
			return U1(cos(phi), sin(phi));
		}
	}
}

inline U1 U1::random(rng_t &rng, double alpha, double alpha2)
{
	assert(alpha >= 0 && alpha2 >= 0);

	// NOTE: this algorithm is only efficient for alpha2 << alpha

	// moderate alpha -> rejection algorithm based on uniform distribution
	if (alpha < 1)
	{
		// approximate distribution of phi (for small alpha)
		std::uniform_real_distribution<double> uni_dist(-M_PI, M_PI);

		while (true)
		{
			++nTries;
			double phi = uni_dist(rng);
			double tr = cos(phi);
			double r = alpha * (tr - 1) + alpha2 * (tr * tr - 1);
			assert(r <= 0);
			if (!std::bernoulli_distribution(exp(r))(rng))
				continue;
			++nAccepts;
			return U1(cos(phi), sin(phi));
		}
	}

	// large alpha -> rejection algorithm based on normal distribution
	else
	{
		// approximate distribution of phi (for larger alpha)
		std::normal_distribution<double> norm_dist(0, 0.5 * M_PI / sqrt(alpha));

		while (true)
		{
			++nTries;
			auto phi = norm_dist(rng);
			if (phi > M_PI || phi < -M_PI)
				continue;
			double tr = cos(phi);
			double r = alpha * tr + alpha2 * tr * tr +
			           alpha * (2 / M_PI / M_PI) * phi * phi - alpha - alpha2;
			assert(r <= 1.0e-8);
			if (!std::bernoulli_distribution(exp(r))(rng))
				continue;
			++nAccepts;
			return U1(cos(phi), sin(phi));
		}
	}
}
#endif

} // namespace mesh
