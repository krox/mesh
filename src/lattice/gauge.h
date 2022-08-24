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
#include "util/hdf5.h"
#include "util/linalg.h"
#include "util/random.h"

namespace mesh {

using util::real;

// Quite often, the Lorentz components are used separately (e.g, in order to
// cshift them into different directions). So it should be beneficial to
// performance to separate them, i.e., not putting the Lorentz index into
// the inner tensor structure.
template <typename vG> using GaugeField = LatticeStack<vG>;

inline util::Stopwatch swRandom, swStaples, swReunitize, swPlaquette;

template <typename T>
void randomGaugeField(Lattice<T> &U, util::xoshiro256 &rng)
{
	util::StopwatchGuard swg(swRandom);
	auto osize = U.grid().osize();
	for (size_t i = 0; i < osize; ++i)
		for (size_t j = 0; j < Lattice<T>::simd_width; ++j)
			vinsert(U.data()[i], j,
			        Lattice<T>::Object::randomGroupElement(rng));
}

template <typename T>
void randomAlgebraField(Lattice<T> &F, util::xoshiro256 &rng)
{
	util::StopwatchGuard swg(swRandom);
	auto osize = F.grid().osize();
	for (size_t i = 0; i < osize; ++i)
		for (size_t j = 0; j < Lattice<T>::simd_width; ++j)
			vinsert(F.data()[i], j,
			        Lattice<T>::Object::randomAlgebraElement(rng));
}

template <typename T>
void randomGaugeField(LatticeStack<T> &U, util::xoshiro256 &rng)
{
	util::StopwatchGuard swg(swRandom);
	for (size_t mu = 0; mu < U.size(); ++mu)
		randomGaugeField(U[mu], rng);
}

template <typename T>
void randomAlgebraField(LatticeStack<T> &F, util::xoshiro256 &rng)
{
	util::StopwatchGuard swg(swRandom);
	for (size_t mu = 0; mu < F.size(); ++mu)
		randomAlgebraField(F[mu], rng);
}

template <typename vG> void reunitize(Lattice<vG> &U)
{
	util::StopwatchGuard swg(swReunitize);
	lattice_apply([](auto &a) { a = projectOnGroupFast(a); }, U);
}

template <typename vG> void reunitize(GaugeField<vG> &U)
{
	util::StopwatchGuard swg(swReunitize);
	lattice_apply([](auto &a) { a = projectOnGroupFast(a); }, U);
}

/** normalized to [0,1] */
template <typename vG> double plaquette(GaugeField<vG> const &U)
{
	util::StopwatchGuard swg(swPlaquette);
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
	util::StopwatchGuard swg(swStaples);
	auto S = Lattice<vG>::zeros(U[0].grid());
	int Nd = U[0].grid().ndim();
	for (int nu = 0; nu < Nd; ++nu)
	{
		if (nu == mu)
			continue;

		// readable version
		// S += cshift(U[nu], mu, 1) * adj(cshift(U[mu], nu, 1)) * adj(U[nu]);
		// S += cshift(adj(cshift(U[nu], mu, 1)) * adj(U[mu]) * U[nu], nu, -1);

		// optimized (fewer temporaries, fewer memory passes)
		auto tmp = cshift(U[nu], mu, 1);
		lattice_apply([](vG &a, vG const &b, vG const &c,
		                 vG const &d) { a += b * adj(d * c); },
		              S, tmp, cshift(U[mu], nu, 1), U[nu]);
		lattice_apply(
		    [](vG &a, vG const &b, vG const &c) { a = adj(b * a) * c; }, tmp,
		    U[mu], U[nu]);
		S += cshift(tmp, nu, -1);
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

// calls f<vG>() with vG determined at runtime based on gauge group and FP prec
template <typename F>
void dispatchByGroup(F f, std::string const &group, int precision)
{
	if (precision == 1)
	{
		if (group == "u1")
			f.template operator()<U1<util::simd<float>>>();
		else if (group == "su2")
			f.template operator()<SU2<util::simd<float>>>();
		else if (group == "su3")
			f.template operator()<SU3<util::simd<float>>>();
		else
			throw std::runtime_error(
			    fmt::format("unknown gauge group '{}'", group));
	}
	else if (precision == 2)
	{
		if (group == "u1")
			f.template operator()<U1<util::simd<double>>>();
		else if (group == "su2")
			f.template operator()<SU2<util::simd<double>>>();
		else if (group == "su3")
			f.template operator()<SU3<util::simd<double>>>();
		else
			throw std::runtime_error(
			    fmt::format("unknown gauge group '{}'", group));
	}
	else
		throw std::runtime_error(
		    fmt::format("invlid precision level '{}'", precision));
}

// if grid is specified, it will be checked to match the file
template <typename vG>
GaugeField<vG> readConfig(std::string const &configName,
                          Grid const *expected_grid = nullptr)
{
	// "my_ensemble.h5/configs/100"
	if (auto p = configName.find(".h5/"); p != std::string::npos)
	{
		auto filename = configName.substr(0, p + 3);
		auto dset = configName.substr(p + 3);
		auto file = util::Hdf5File::open(filename);
		auto geom = file.get_attribute<std::vector<int>>("geometry");
		auto &grid = Grid::make(Coordinate(geom.begin(), geom.end()),
		                        (int)GaugeField<vG>::simd_width);
		if (file.get_attribute<std::string>("group") != vG::name())
			throw std::runtime_error(fmt::format(
			    "group mismatch on load. Expected {}, got {}\n", vG::name(),
			    file.get_attribute<std::string>("group")));
		if (expected_grid && expected_grid != &grid)
			throw std::runtime_error(
			    fmt::format("grid mismatch on load. Expected {}, got {}\n",
			                expected_grid->to_string(), grid.to_string()));
		auto U = GaugeField<vG>(grid);
		readFromFile(file, dset, U);
		return U;
	}
	else
		throw std::runtime_error(
		    fmt::format("unknown config file format '{}'", configName));
}

} // namespace mesh
