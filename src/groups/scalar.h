#ifndef GROUPS_SCALAR_H
#define GROUPS_SCALAR_H

#include <array>
#include <cassert>
#include <random>

#include "util/random.h"

template <size_t N> struct Scalar
{
	std::array<double, N> v;
	static constexpr size_t repSize() { return N; }

	double &operator[](size_t i) { return v[i]; }
	const double &operator[](size_t i) const { return v[i]; }

	/** constructors */
	Scalar() = default;

	/** special elements */
	static Scalar zero()
	{
		Scalar s;
		s.v.fill(0);
		return s;
	}
	static Scalar one()
	{
		Scalar s;
		s.v.fill(0);
		s[0] = 1;
		return s;
	}

	/** random group element */
	static Scalar randomSphere(rng_t &rng)
	{
		std::normal_distribution<double> norm_dist(0.0, 1.0);

		Scalar r;
		r[0] = norm_dist(rng);
		r[1] = norm_dist(rng);
		r[2] = norm_dist(rng);
		return r.normalize();
	}

	static Scalar randomSphere(rng_t &rng, double alpha)
	{
		assert(N == 3 && "not implemented for N != 3");
		assert(alpha > 0.0);
		std::uniform_real_distribution<double> uni_dist(exp(-2 * alpha), 1.0);
		std::normal_distribution<double> norm_dist(0.0, 1.0);

		Scalar r;

		// exponential distribution of r[0] (only true for N = 3 !)
		r[0] = log(uni_dist(rng)) / alpha + 1;

		// uniform sphere distribution of remaining entries
		r[1] = norm_dist(rng);
		r[2] = norm_dist(rng);
		double s = sqrt((1 - r[0] * r[0]) / (r[1] * r[1] + r[2] * r[2]));
		r[1] *= s;
		r[2] *= s;
		return r;
	}

	static Scalar randomNormal(rng_t &rng);

	/** vector-space operations */
	Scalar operator*(double b) const
	{
		Scalar s;
		for (size_t i = 0; i < N; ++i)
			s[i] = (*this)[i] * b;
		return s;
	}
	Scalar operator/(double b) const
	{
		Scalar s;
		for (size_t i = 0; i < N; ++i)
			s[i] = (*this)[i] / b;
		return s;
	}
	Scalar operator+(const Scalar &b) const
	{
		Scalar s;
		for (size_t i = 0; i < N; ++i)
			s[i] = (*this)[i] + b[i];
		return s;
	}
	Scalar operator-(const Scalar &b) const
	{
		Scalar s;
		for (size_t i = 0; i < N; ++i)
			s[i] = (*this)[i] - b[i];
		return s;
	}

	/** convenience */
	void operator*=(double b) { *this = *this * b; }
	void operator/=(double b) { *this = *this / b; }
	void operator+=(const Scalar &b) { *this = *this + b; }
	void operator-=(const Scalar &b) { *this = *this - b; }

	/** sclar product */
	double dot(const Scalar &b) const
	{
		double s = 0.0;
		for (size_t i = 0; i < N; ++i)
			s += (*this)[i] * b[i];
		return s;
	}

	/** misc */
	double norm() const { return sqrt(this->dot(*this)); }
	Scalar normalize() const { return *this / norm(); }
};

#endif
