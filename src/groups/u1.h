#ifndef GROUPS_U1_H
#define GROUPS_U1_H

#include <cmath>
#include <random>

#include "util/random.h"

/** unitary group U(1) */
struct U1
{
	// represents v0 + i*v1
	double v[2];

	static constexpr size_t repSize() { return 2; }

	double &operator[](size_t i) { return v[i]; }
	const double &operator[](size_t i) const { return v[i]; }

	/** constructors */
	U1() = default;
	explicit U1(double a, double b) : v{a, b} {}

	/** random group element */
	static U1 random(rng_t &rng);
	static U1 random(rng_t &rng, double alpha);
	static U1 random(rng_t &rng, double alpha, double alpha2);

	/** special elements */
	static U1 zero() { return U1(0, 0); }
	static U1 one() { return U1(1, 0); }

	/** (unnormalized) probability distribution */
	static double dist(double alpha, double alpha2, double tr)
	{
		return exp(alpha * tr + alpha2 * tr * tr) * pow(1 - tr * tr, -0.5);
	}

	/** scalar operators */
	U1 operator*(double b) const { return U1(v[0] * b, v[1] * b); }
	U1 operator/(double b) const { return U1(v[0] / b, v[1] / b); }
	void operator*=(double b) { *this = *this * b; }
	void operator/=(double b) { *this = *this / b; }

	/** group operators */
	U1 operator*(const U1 &b) const
	{
		U1 r;
		r[0] = v[0] * b[0] - v[1] * b[1];
		r[1] = v[0] * b[1] + v[1] * b[0];
		return r;
	}
	U1 operator+(const U1 &b) const { return U1(v[0] + b[0], v[1] + b[1]); }
	U1 operator-(const U1 &b) const { return U1(v[0] - b[0], v[1] - b[1]); }

	void operator*=(const U1 &b) { *this = *this * b; }
	void operator+=(const U1 &b) { *this = *this + b; }
	void operator-=(const U1 &b) { *this = *this - b; }

	/** misc */
	U1 adjoint() const { return U1(v[0], -v[1]); }

	double norm() const { return sqrt(v[0] * v[0] + v[1] * v[1]); }

	U1 normalize() const { return *this / norm(); }

	/** distance from group */
	double error() const { return std::fabs(norm() - 1.0); }

	double action() const { return v[0]; }

	U1 traceless() const { return U1(0, 0); }
	U1 algebra() const { return U1(0, v[1]); }
	U1 sym() const { return U1(v[0], 0); }
	U1 antisym() const { return U1(0, v[1]); }

	/** statistics on random element generation */
	static inline uint64_t nAccepts = 0, nTries = 0;
	static void clearStats()
	{
		nAccepts = 0;
		nTries = 0;
	}
	static double accProb() { return (double)nAccepts / nTries; }
};

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
