#ifndef GROUPS_Z2_H
#define GROUPS_Z2_H

#include <random>

#include "util/random.h"

struct Z2
{
	double a;

	static constexpr size_t repSize() { return 1; }

	/** constructors */
	Z2() = default;
	explicit Z2(double a) : a(a) {}

	/** random group element */
	static Z2 random(rng_t &rng);
	static Z2 random(rng_t &rng, double alpha);
	static Z2 random(rng_t &rng, double alpha, double alpha2);

	/** special elements */
	static Z2 zero() { return Z2(0); }
	static Z2 one() { return Z2(1); }

	/** scalar operators */
	Z2 operator*(double b) const { return Z2(a * b); }
	Z2 operator/(double b) const { return Z2(a / b); }
	void operator*=(double b) { *this = *this * b; }
	void operator/=(double b) { *this = *this / b; }

	/** group operators */
	Z2 operator*(const Z2 &b) const { return Z2(a * b.a); }
	Z2 operator+(const Z2 &b) const { return Z2(a + b.a); }
	Z2 operator-(const Z2 &b) const { return Z2(a - b.a); }
	void operator*=(const Z2 &b) { *this = *this * b; }
	void operator+=(const Z2 &b) { *this = *this + b; }
	void operator-=(const Z2 &b) { *this = *this - b; }

	/** misc */
	Z2 adjoint() const { return Z2(a); }
	double norm() const { return fabs(a); }
	Z2 normalize() const { return a < 0 ? Z2(-1) : Z2(1); }
	double error() const { return std::fabs(norm() - 1.0); }

	double action() const { return a; }
	Z2 traceless() const { return Z2(0); }
	Z2 algebra() const { return Z2(0); }
	Z2 sym() const { return Z2(a); }
	Z2 antisym() const { return Z2(0); }

	/** statistics on random element generation */
	static void clearStats() {}
	static double accProb() { return 1.0; }
};

inline Z2 Z2::random(rng_t &rng)
{
	return std::bernoulli_distribution(0.5)(rng) ? Z2(1) : Z2(-1);
}

inline Z2 Z2::random(rng_t &rng, double alpha)
{
	double p = exp(alpha);
	double q = exp(-alpha);
	return std::bernoulli_distribution(p / (p + q))(rng) ? Z2(1) : Z2(-1);
}

inline Z2 Z2::random(rng_t &rng, double alpha, [[maybe_unused]] double alpha2)
{
	// there is no secondary action in Z2, so ignoring alpha2 is correct
	return random(rng, alpha);
}

#endif
