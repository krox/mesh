#ifndef MESH_SU2_H
#define MESH_SU2_H

#include <cmath>
#include <random>

/** Special unitary group SU(2) */
struct SU2
{
	// represents v0 + i*v_k*sigma_k
	// which is equal to  v0 + i v3   v2 + i v1
	//                   -v2 + i v1   v0 - i v3
	double v[4];

	static constexpr size_t repSize() { return 4; }

	double &operator[](size_t i) { return v[i]; }
	const double &operator[](size_t i) const { return v[i]; }

	/** constructors */
	SU2() = default;
	explicit SU2(double a, double b, double c, double d) : v{a, b, c, d} {}

	/** random group element */
	static SU2 random(rng_t &rng);
	static SU2 random(rng_t &rng, double alpha);
	static SU2 random(rng_t &rng, double alpha, double alpha2);

	/** special elements */
	static SU2 zero() { return SU2(0, 0, 0, 0); }
	static SU2 one() { return SU2(1, 0, 0, 0); }

	/** scalar operators */
	SU2 operator*(double b) const
	{
		return SU2(v[0] * b, v[1] * b, v[2] * b, v[3] * b);
	}
	SU2 operator/(double b) const
	{
		return SU2(v[0] / b, v[1] / b, v[2] / b, v[3] / b);
	}
	void operator*=(double b) { *this = *this * b; }
	void operator/=(double b) { *this = *this / b; }

	/** group operators */
	SU2 operator*(const SU2 &b) const
	{
		SU2 r;
		r[0] = v[0] * b[0] - v[1] * b[1] - v[2] * b[2] - v[3] * b[3];
		r[1] = v[0] * b[1] + v[1] * b[0] - v[2] * b[3] + v[3] * b[2];
		r[2] = v[0] * b[2] + v[2] * b[0] - v[3] * b[1] + v[1] * b[3];
		r[3] = v[0] * b[3] + v[3] * b[0] - v[1] * b[2] + v[2] * b[1];
		return r;
	}
	SU2 operator+(const SU2 &b) const
	{
		return SU2(v[0] + b[0], v[1] + b[1], v[2] + b[2], v[3] + b[3]);
	}
	SU2 operator-(const SU2 &b) const
	{
		return SU2(v[0] - b[0], v[1] - b[1], v[2] - b[2], v[3] - b[3]);
	}

	void operator*=(const SU2 &b) { *this = *this * b; }
	void operator+=(const SU2 &b) { *this = *this + b; }
	void operator-=(const SU2 &b) { *this = *this - b; }

	/** misc */
	SU2 adjoint() const { return SU2(v[0], -v[1], -v[2], -v[3]); }

	double norm() const
	{
		return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
	}

	SU2 normalize() const { return *this / norm(); }

	double action() const { return v[0]; }

	/** NOTE: this has the wrong phase factor */
	SU2 algebra() const { return SU2(0, v[1], v[2], v[3]); }

	/** statistics on random element generation */
	static inline uint64_t nAccepts = 0, nTries = 0;
	static void clearStats()
	{
		nAccepts = 0;
		nTries = 0;
	}
	static double accProb() { return (double)nAccepts / nTries; }
};

inline SU2 SU2::random(rng_t &rng)
{
	std::normal_distribution d;
	return SU2(d(rng), d(rng), d(rng), d(rng)).normalize();
}

inline SU2 SU2::random(rng_t &rng, double alpha)
{
	assert(alpha >= 0);

	// very small alpha -> just sample uniformly
	if (alpha < 1.0e-8)
		return random(rng);

	// moderate alpha -> basic rejection algorithm
	else if (alpha <= 1.5)
	{
		// naive rejection algorithm
		while (true)
		{
			++nTries;
			auto r = random(rng);
			double p = std::min(exp(alpha * (r[0] - 1)), 1.0);
			if (!std::bernoulli_distribution(p)(rng))
				continue;
			++nAccepts;
			return r;
		}
	}

	// large alpha -> advanced algorithm by A.D.Kennedy and B.J.Pendleton
	else
	{
		std::exponential_distribution<double> exp_dist(alpha);
		std::normal_distribution<double> norm_dist;
		std::uniform_real_distribution<double> uni_dist;
		while (true)
		{
			++nTries;
			double x = exp_dist(rng);
			double x2 = exp_dist(rng);
			double c = pow(cos(uni_dist(rng) * 2 * M_PI), 2);
			double d = x2 + x * c;

			if (pow(uni_dist(rng), 2) > 1 - 0.5 * d)
				continue;

			SU2 r;
			r[0] = 1 - d;
			r[1] = norm_dist(rng);
			r[2] = norm_dist(rng);
			r[3] = norm_dist(rng);
			double f = sqrt((1 - r[0] * r[0]) /
			                (r[1] * r[1] + r[2] * r[2] + r[3] * r[3]));
			r[1] *= f;
			r[2] *= f;
			r[3] *= f;
			++nAccepts;
			return r;
		}
	}
}

inline SU2 SU2::random(rng_t &rng, double alpha, double alpha2)
{
	assert(alpha >= 0 && alpha2 >= 0);

	assert(alpha2 == 0);
	return random(rng, alpha);
}

#endif
