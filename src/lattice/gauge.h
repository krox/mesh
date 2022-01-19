#pragma once

/**
 * Utilities for lattice gauge theory.
 * Might be split into multiple files in the future.
 */

#include "groups/su2.h"
#include "groups/su3.h"
#include "groups/u1.h"
#include "lattice/lattice.h"
#include "util/complex.h"
#include "util/linalg.h"
#include "util/random.h"
#include <vector>

namespace mesh {

using util::real;

// TODO: We should use a different container than std::vector here. Its not
//       very convenient to explicit overload all operations to handle the
//       Lorentz-parallel case. And operator-overloading (in lattice.h) is
//       probably even forbidden by the C++ standard.
template <typename vG> using GaugeField = std::vector<Lattice<vG>>;

template <typename vG> GaugeField<vG> makeGaugeField(Grid const &g)
{
	GaugeField<vG> r;
	for (int mu = 0; mu < g.ndim(); ++mu)
		r.emplace_back(g);
	return r;
}

template <typename T>
void randomGaugeField(Lattice<T> &U, util::xoshiro256 &rng)
{
	auto osize = U.grid().osize();
	for (size_t i = 0; i < osize; ++i)
		for (size_t j = 0; j < TensorTraits<T>::simdWidth; ++j)
			vinsert(U.data()[i], j,
			        TensorTraits<T>::ScalarType::randomGroupElement(rng));
}

template <typename T>
void randomAlgebraField(Lattice<T> &F, util::xoshiro256 &rng)
{
	auto osize = F.grid().osize();
	for (size_t i = 0; i < osize; ++i)
		for (size_t j = 0; j < TensorTraits<T>::simdWidth; ++j)
			vinsert(F.data()[i], j,
			        TensorTraits<T>::ScalarType::randomAlgebraElement(rng));
}

template <typename T>
void randomGaugeField(std::vector<T> &U, util::xoshiro256 &rng)
{
	for (auto &Umu : U)
		randomGaugeField(Umu, rng);
}

template <typename T>
void randomAlgebraField(std::vector<T> &F, util::xoshiro256 &rng)
{
	for (auto &Fmu : F)
		randomAlgebraField(Fmu, rng);
}

template <typename vG> void reunitize(GaugeField<vG> &U)
{
	size_t osites = U[0].grid().osize();
	for (auto &Umu : U)
		for (size_t i = 0; i < osites; ++i)
			Umu.data()[i] = projectOnGroup(Umu.data()[i]);
}

/** normalized to [0,1] */
template <typename vG> double plaquette(GaugeField<vG> const &U)
{
	double vol = U[0].grid().size();
	double s = 0;
	int Nd = U[0].grid().ndim();
	for (int mu = 0; mu < Nd; ++mu)
		for (int nu = mu + 1; nu < Nd; ++nu)
		{
			auto tmp = U[mu] * cshift(U[nu], mu, 1) *
			           adj(cshift(U[mu], nu, 1)) * adj(U[nu]);
			s += real(sumTrace(tmp));
		}
	return s / (vol * Nd * (Nd - 1) * 0.5) / vG::Nc();
}

/**
 * Compute sum of 2*Nd staples written such that
 * plaquette(U) = const * sum_mu Real Trace U[mu] * staple(U, mu)
 */
template <typename vG> Lattice<vG> stapleSum(GaugeField<vG> const &U, int mu)
{
	auto S = Lattice<vG>::zeros(U[0].grid());
	int Nd = U[0].grid().ndim();
	for (int nu = 0; nu < Nd; ++nu)
	{
		if (nu == mu)
			continue;
		S += cshift(U[nu], mu, 1) * adj(cshift(U[mu], nu, 1)) * adj(U[nu]);
		S += cshift(adj(cshift(U[nu], mu, 1)) * adj(U[mu]) * U[nu], nu, -1);
	}
	return S;
}

template <typename vG> double wilsonAction(GaugeField<vG> const &U, double beta)
{
	int Nd = U[0].grid().ndim();
	double vol = U[0].grid().size();
	return (1.0 - plaquette(U)) * (beta * Nd * (Nd - 1) * 0.5 * vol);
}

template <typename vG>
Lattice<vG> wilsonDeriv(GaugeField<vG> const &U, int mu, double beta)
{
	return projectOnAlgebra(U[mu] * stapleSum(U, mu) * (beta / (2 * vG::Nc())));
}

} // namespace mesh
