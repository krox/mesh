#ifndef MESH_Z2_H
#define MESH_Z2_H

#include <random>

struct Z2
{
	double a;

	/** constructors */
	Z2() = default;
	explicit Z2(double a) : a(a) {}

	/** random group element */
	template <typename Rng> static Z2 random(Rng &rng);
	template <typename Rng> static Z2 random(Rng &rng, double alpha);
	template <typename Rng>
	static Z2 random(Rng &rng, double alpha, double alpha2);

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
	double action() const { return a; }
	Z2 algebra() const { return Z2(0); }
};

template <typename Rng> inline Z2 Z2::random(Rng &rng)
{
	return std::bernoulli_distribution(0.5)(rng) ? Z2(1) : Z2(-1);
}

template <typename Rng> inline Z2 Z2::random(Rng &rng, double alpha)
{
	double p = exp(alpha);
	double q = exp(-alpha);
	return std::bernoulli_distribution(p / (p + q))(rng) ? Z2(1) : Z2(-1);
}

template <typename Rng>
inline Z2 Z2::random(Rng &rng, double alpha, [[maybe_unused]] double alpha2)
{
	// there is no secondary action in Z2, so ignoring alpha2 is correct
	return random(rng, alpha);
}

#endif
