#ifndef GROUPS_SU3_H
#define GROUPS_SU3_H

#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <unsupported/Eigen/MatrixFunctions>

#include "groups/su2.h"
#include "util/random.h"

/** Special unitary group SU(3) */
struct SU3
{
	Eigen::Matrix3cd v;

	static constexpr size_t repSize() { return 18; }

	/** constructors */
	SU3() = default;
	explicit SU3(const Eigen::Matrix3cd &v) : v(v) {}

	/** random group element */
	template <typename Rng> static SU3 random(Rng &rng);
	// template <typename Rng> static SU3 random(Rng &rng, double alpha);

	/** special elements */
	static SU3 zero() { return SU3(Eigen::Matrix3cd::Zero()); }
	static SU3 one() { return SU3(Eigen::Matrix3cd::Identity()); }

	/** scalar operators */
	SU3 operator*(double b) const { return SU3(v * b); }
	SU3 operator/(double b) const { return SU3(v / b); }
	void operator*=(double b) { *this = *this * b; }
	void operator/=(double b) { *this = *this / b; }

	/** group operators */
	SU3 operator*(const SU3 &b) const { return SU3(v * b.v); }
	SU3 operator+(const SU3 &b) const { return SU3(v + b.v); }
	SU3 operator-(const SU3 &b) const { return SU3(v - b.v); }
	void operator*=(const SU3 &b) { *this = *this * b; }
	void operator+=(const SU3 &b) { *this = *this + b; }
	void operator-=(const SU3 &b) { *this = *this - b; }

	/** SU(2) subgroups */
	SU2 sub1() const
	{
		// 2x2 sub-matrix
		std::complex<double> a = v(0, 0);
		std::complex<double> b = v(0, 1);
		std::complex<double> c = v(1, 0);
		std::complex<double> d = v(1, 1);

		// project to multiples of SU(2)
		return SU2(0.5 * (a.real() + d.real()), 0.5 * (b.imag() + c.imag()),
		           0.5 * (b.real() - c.real()), 0.5 * (a.imag() - d.imag()));
	}

	SU2 sub2() const
	{
		// 2x2 sub-matrix
		std::complex<double> a = v(1, 1);
		std::complex<double> b = v(1, 2);
		std::complex<double> c = v(2, 1);
		std::complex<double> d = v(2, 2);

		// project to multiples of SU(2)
		return SU2(0.5 * (a.real() + d.real()), 0.5 * (b.imag() + c.imag()),
		           0.5 * (b.real() - c.real()), 0.5 * (a.imag() - d.imag()));
	}

	SU2 sub3() const
	{
		// 2x2 sub-matrix
		std::complex<double> a = v(0, 0);
		std::complex<double> b = v(0, 2);
		std::complex<double> c = v(2, 0);
		std::complex<double> d = v(2, 2);

		// project to multiples of SU(2)
		return SU2(0.5 * (a.real() + d.real()), 0.5 * (b.imag() + c.imag()),
		           0.5 * (b.real() - c.real()), 0.5 * (a.imag() - d.imag()));
	}

	/** compute a * this, for b in subgroup */
	SU3 leftMul1(SU2 a)
	{
		SU3 sub = SU3::one();
		sub.v(0, 0) = std::complex<double>(a[0], a[3]);
		sub.v(0, 1) = std::complex<double>(a[2], a[1]);
		sub.v(1, 0) = std::complex<double>(-a[2], a[1]);
		sub.v(1, 1) = std::complex<double>(a[0], -a[3]);
		return sub * *this;
	}

	SU3 leftMul2(SU2 a)
	{
		SU3 sub = SU3::one();
		sub.v(1, 1) = std::complex<double>(a[0], a[3]);
		sub.v(1, 2) = std::complex<double>(a[2], a[1]);
		sub.v(2, 1) = std::complex<double>(-a[2], a[1]);
		sub.v(2, 2) = std::complex<double>(a[0], -a[3]);
		return sub * *this;
	}

	SU3 leftMul3(SU2 a)
	{
		SU3 sub = SU3::one();
		sub.v(0, 0) = std::complex<double>(a[0], a[3]);
		sub.v(0, 2) = std::complex<double>(a[2], a[1]);
		sub.v(2, 0) = std::complex<double>(-a[2], a[1]);
		sub.v(2, 2) = std::complex<double>(a[0], -a[3]);
		return sub * *this;
	}

	/** misc */
	SU3 adjoint() const { return SU3(v.adjoint()); }

	double norm() const { return v.norm(); } // Frobenius norm

	/** projection to SU(3) (slow) */
	SU3 normalize() const
	{
		SU3 r;
		// NOTE: .pow(-0.5) is much slower than .inverse().sqrt()
		r.v = v * (v.adjoint() * v).inverse().sqrt();
		r.v *= std::pow(r.v.determinant(), -1.0 / 3);
		return r;
	}

	/** approximate projection to SU(3) (faster, valid if already close) */
	SU3 normalizeFast() const
	{
		SU3 r;
		r.v = v * 1.5 - v * v.adjoint() * v * 0.5;
		r.v *= 1.0 - (1 / 3.0) * (r.v.determinant() - 1.0);
		return r;
	}

	double action() const { return (1.0 / 3.0) * v.trace().real(); }

	SU3 algebra() const
	{
		SU3 a = *this;
		a = (a + a.adjoint()) * 0.5;
		a -= one() * (a.v.trace().real() / 3.0);
		return a;
	}

	SU3 traceless() const
	{
		SU3 a = *this;
		a -= one() * (a.v.trace().real() / 3.0);
		return a;
	}

	SU3 sym() const { return (*this + this->adjoint()) * 0.5; }
	SU3 antisym() const { return (*this - this->adjoint()) * 0.5; }

	/** statistics on random element generation */
	static inline uint64_t nAccepts = 0, nTries = 0;
	static void clearStats()
	{
		nAccepts = 0;
		nTries = 0;
	}
	static double accProb() { return (double)nAccepts / nTries; }
};

template <typename Rng> SU3 SU3::random(Rng &rng)
{
	// Idea: * create normal distributed entries
	//       * compute QR decomposition
	//       * Q is now uniform in U(3)
	std::normal_distribution d;
	Eigen::Matrix3cd m;
	for (size_t i = 0; i < 3; ++i)
		for (size_t j = 0; j < 3; ++j)
			m(i, j) = std::complex(d(rng), d(rng));
	m = m.householderQr().householderQ();
	m.col(0) /= m.determinant();
	return SU3(m);
}

#endif
