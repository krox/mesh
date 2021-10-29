#pragma once

/**
 * Utilities for lattice gauge theory.
 * Might be split into multiple files in the future.
 */

#include "groups/su3.h"
#include "lattice/lattice.h"
#include "util/complex.h"
#include "util/linalg.h"
#include "util/ndarray.h"
#include "util/random.h"
#include "util/span.h"

namespace mesh {

namespace QCD {

using Real = double;
using vReal = util::simd<double>;
using Complex = util::complex<Real>;
using vComplex = util::complex<vReal>;

static constexpr int Nd = 4;
static constexpr int Nc = 3;

using ColorMatrix = SU3<Real>;
using vColorMatrix = SU3<vReal>;

using LatticeColorMatrix = Lattice<vColorMatrix>;
using GaugeField = std::array<LatticeColorMatrix, Nd>;

GaugeField randomGaugeField(Grid const &g, util::xoshiro256 &rng)
{
	GaugeField U;
	for (int mu = 0; mu < Nd; ++mu)
	{
		U[mu] = LatticeColorMatrix(g);
		for (size_t i = 0; i < g.osize(); ++i)
			for (size_t j = 0; j < TensorTraits<vColorMatrix>::simdWidth; ++j)
				vinsert(U[mu].data()[i], j,
				        ColorMatrix::randomGroupElement(rng));
	}
	return U;
}

GaugeField randomAlgebraField(Grid const &g, util::xoshiro256 &rng)
{
	GaugeField F;
	for (int mu = 0; mu < Nd; ++mu)
	{
		F[mu] = LatticeColorMatrix(g);
		for (size_t i = 0; i < g.osize(); ++i)
			for (size_t j = 0; j < TensorTraits<vColorMatrix>::simdWidth; ++j)
				vinsert(F[mu].data()[i], j,
				        ColorMatrix::randomAlgebraElement(rng));
	}
	return F;
}

void reunitize(GaugeField &U)
{
	size_t osites = U[0].grid().osize();
	for (int mu = 0; mu < Nd; ++mu)
		for (size_t i = 0; i < osites; ++i)
			U[mu].data()[i] = projectOnGroup(U[mu].data()[i]);
}

/** normalized to [0,1] */
double plaquette(GaugeField const &U)
{
	double vol = U[0].grid().size();
	double s = 0;
	for (int mu = 0; mu < Nd; ++mu)
		for (int nu = mu + 1; nu < Nd; ++nu)
		{
			auto tmp = U[mu] * cshift(U[nu], mu, 1) *
			           adj(cshift(U[mu], nu, 1)) * adj(U[nu]);
			s += real(sumTrace(tmp));
		}
	return s / (vol * Nd * (Nd - 1) / 2) / Nc;
}
/**
 * Compute sum of 2*Nd staples written such that
 * plaquette(U) = const * sum_mu Real Trace U[mu] * staple(U, mu)
 */
LatticeColorMatrix stapleSum(GaugeField const &U, int mu)
{
	auto S = LatticeColorMatrix::zeros(U[0].grid());
	for (int nu = 0; nu < Nd; ++nu)
	{
		if (nu == mu)
			continue;
		S += cshift(U[nu], mu, 1) * adj(cshift(U[mu], nu, 1)) * adj(U[nu]);
		S += cshift(adj(cshift(U[nu], mu, 1)) * adj(U[mu]) * U[nu], nu, -1);
	}
	return S;
}

double wilsonAction(GaugeField const &U, double beta)
{
	double vol = U[0].grid().size();
	return (1.0 - plaquette(U)) * (beta * Nd * (Nd - 1) / 2 * vol);
}

LatticeColorMatrix wilsonDeriv(GaugeField const &U, int mu, double beta)
{
	return projectOnAlgebra(U[mu] * stapleSum(U, mu) * (beta / (2 * Nc)));
}

} // namespace QCD

} // namespace mesh
