#pragma once

#include "util/random.h"
#include <cmath>

namespace mesh {

// Special unitary group SU(2), stored with just 4 real numbers
template <typename T> struct SU2
{
	// human-readable name of the group
	static constexpr std::string_view name() { return "SU(2)"; }

	// dimension of the Lie group
	//   = number of generators
	//   = dimension of adjoint representation
	static constexpr int dim() { return 3; }

	// number of "colors"
	//   = dimension of defining/fundamental representation
	static constexpr int Nc() { return 2; }

	// represents v0 + i*v_k*sigma_k
	// which is equal to  v0 + i v3   v2 + i v1
	//                   -v2 + i v1   v0 - i v3
	// the generators are
	// iT^1 = (0, 1/2, 0, 0)
	// iT^2 = (0, 0, 1/2, 0)
	// iT^3 = (0, 0, 0, 1/2)
	std::array<T, 4> v_;

	T &operator[](int i) { return v_[i]; }
	T const &operator[](int i) const { return v_[i]; }

	// constructors
	SU2() = default;
	explicit SU2(T a) : v_{a, 0, 0, 0} {}
	SU2(T a, T b, T c, T d) : v_{a, b, c, d} {}
	static SU2 zero() { return SU2(T(0), T(0), T(0), T(0)); }
	static SU2 one() { return SU2(T(1), T(0), T(0), T(0)); }
	template <typename Rng> static SU2 randomGroupElement(Rng &rng)
	{
		return projectOnGroup(
		    SU2{rng.normal(), rng.normal(), rng.normal(), rng.normal()});
	}
	template <typename Rng> static SU2 randomAlgebraElement(Rng &rng)
	{
		return {0, rng.normal() * 0.5, rng.normal() * 0.5, rng.normal() * 0.5};
	}
};

// Unary SU2

template <typename T> SU2<T> operator-(SU2<T> const &a)
{
	return {-a[0], -a[1], -a[2], -a[3]};
}

template <typename T> SU2<T> adj(SU2<T> const &a)
{
	return {a[0], -a[1], -a[2], -a[3]};
}

// trace in defining representation
template <typename T> T trace(SU2<T> const &a)
{
	// trace of SU(2) matrices is always real. This is not true for larger SU(N)
	return a[0] * 2.0;
}

template <typename T> T norm2(SU2<T> const &a)
{
	return 2.0 * (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]);
}

template <typename T> SU2<T> exp(SU2<T> const &a)
{
	// exact version (only valid for a[0] = 0)
	/*
	auto alpha = sqrt(a[1] * a[1] + a[2] * a[2] + a[3] * a[3]);
	auto f = sin(alpha) / alpha;
	SU2<T> r;
	r[0] = cos(alpha);
	r[1] = a[1] * f;
	r[2] = a[2] * f;
	r[3] = a[3] * f;
	*/

	// explicit taylor version
	auto b = a * (1.0 / 16.0);
	auto inc = b;
	auto r = SU2<T>::one() + b;
	for (int n = 2; n <= 12; ++n)
	{
		inc = inc * b * (1.0 / n);
		r += inc;
	}
	for (int i = 0; i < 4; ++i)
		r = r * r;
	return r;
}

template <typename T> SU2<T> projectOnGroup(SU2<T> const &a)
{
	auto s = a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3];

	return a * (1.0 / sqrt(s)); // exact version
	// return a * (1.5 - 0.5 * s); // fast, but only if a is already close
}

template <typename T> SU2<T> projectOnAlgebra(SU2<T> const &a)
{
	return {T(0), a[1], a[2], a[3]};
}

// binary SU2 <-> SU2

template <typename T> SU2<T> operator+(SU2<T> const &a, SU2<T> const &b)
{
	return {a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
}

template <typename T> SU2<T> operator-(SU2<T> const &a, SU2<T> const &b)
{
	return {a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]};
}

template <typename T> SU2<T> operator*(SU2<T> const &a, SU2<T> const &b)
{
	return {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
	        a[0] * b[1] + a[1] * b[0] - a[2] * b[3] + a[3] * b[2],
	        a[0] * b[2] + a[2] * b[0] - a[3] * b[1] + a[1] * b[3],
	        a[0] * b[3] + a[3] * b[0] - a[1] * b[2] + a[2] * b[1]};
}

template <typename T> void operator+=(SU2<T> &a, SU2<T> const &b)
{
	a[0] += b[0];
	a[1] += b[1];
	a[2] += b[2];
	a[3] += b[3];
}

template <typename T> void operator-=(SU2<T> &a, SU2<T> const &b)
{
	a[0] -= b[0];
	a[1] -= b[1];
	a[2] -= b[2];
	a[3] -= b[3];
}

template <typename T> void operator*=(SU2<T> &a, SU2<T> const &b) { a = a * b; }

// binary SU2 <-> scalar

template <typename T> SU2<T> operator*(SU2<T> const &a, T b)
{
	return {a[0] * b, a[1] * b, a[2] * b, a[3] * b};
}

template <typename T, std::enable_if_t<!std::is_same_v<T, double>, bool> = true>
SU2<T> operator*(SU2<T> const &a, double b)
{
	return {a[0] * b, a[1] * b, a[2] * b, a[3] * b};
}

template <typename T> void operator*=(SU2<T> &a, T b)
{
	a[0] *= b;
	a[1] *= b;
	a[2] *= b;
	a[3] *= b;
}

template <typename T> void operator*=(SU2<T> &a, double b)
{
	a[0] *= b;
	a[1] *= b;
	a[2] *= b;
	a[3] *= b;
}

template <typename T> auto vsum(SU2<T> const &a)
{
	return SU2(vsum(a[0]), vsum(a[1]), vsum(a[2]), vsum(a[3]));
}
template <typename T> auto vextract(SU2<T> const &a, size_t lane)
{
	return SU2(vextract(a[0], lane), vextract(a[1], lane), vextract(a[2], lane),
	           vextract(a[3], lane));
}
template <typename T, typename U>
void vinsert(SU2<T> &a, size_t lane, SU2<U> const &b)
{
	vinsert(a[0], lane, b[0]);
	vinsert(a[1], lane, b[1]);
	vinsert(a[2], lane, b[2]);
	vinsert(a[3], lane, b[3]);
}

template <typename> struct TensorTraits;
template <typename T> struct TensorTraits<SU2<T>>
{
	using ScalarType = SU2<typename TensorTraits<T>::ScalarType>;
	static constexpr size_t simdWidth = TensorTraits<T>::simdWidth;
	using BaseType = typename TensorTraits<T>::BaseType;
};

/*
// SU2 heatbath
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
*/

} // namespace mesh
