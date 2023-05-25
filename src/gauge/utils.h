#pragma once

/**
 * Utilities for lattice gauge theory.
 * Might be split into multiple files in the future.
 */

#include "gauge/groups.h"
#include "lattice/lattice.h"
#include "util/complex.h"
#include "util/hdf5.h"
#include "util/linalg.h"
#include "util/random.h"

namespace mesh {

// Quite often, the Lorentz components are used separately (e.g, in order to
// cshift them into different directions). So it should be beneficial to
// performance to separate them, i.e., not putting the Lorentz index into
// the inner tensor structure.
template <class G> using GaugeField = LatticeStack<G>;

inline util::Stopwatch swRandom, swStaples, swReunitize, swPlaquette;

using util::real;

// lattice versions of some gauge group operations
UTIL_DEFINE_LATTICE_REDUCTION(norm2, norm2)
UTIL_DEFINE_LATTICE_REDUCTION(sum_real_trace, gauge::real_trace)
UTIL_DEFINE_LATTICE_UNARY(real_trace, real_trace)
UTIL_DEFINE_LATTICE_UNARY(adj, adj)
UTIL_DEFINE_LATTICE_UNARY(exp, gauge::exp)
UTIL_DEFINE_LATTICE_UNARY(project_on_algebra, project_on_algebra)

template <typename T>
void random_gauge_field(Lattice<T> &U, util::xoshiro256 &rng)
{
	// TODO: actually parallelize this
	util::StopwatchGuard swg(swRandom);
	auto s = U.grid().size();
	for (size_t i = 0; i < s; ++i)
		gauge::random_group_element(U.data()[i], rng);
}

template <typename T>
void random_algebra_field(Lattice<T> &F, util::xoshiro256 &rng)
{
	util::StopwatchGuard swg(swRandom);
	auto s = F.grid().size();
	for (size_t i = 0; i < s; ++i)
		gauge::random_algebra_element(F.data()[i], rng);
}

template <typename T>
void random_gauge_field(LatticeStack<T> &U, util::xoshiro256 &rng)
{
	util::StopwatchGuard swg(swRandom);
	for (size_t mu = 0; mu < U.size(); ++mu)
		random_gauge_field(U[mu], rng);
}

template <typename T>
void random_algebra_field(LatticeStack<T> &F, util::xoshiro256 &rng)
{
	util::StopwatchGuard swg(swRandom);
	for (size_t mu = 0; mu < F.size(); ++mu)
		random_algebra_field(F[mu], rng);
}

template <typename vG> void reunitize(Lattice<vG> &U)
{
	util::StopwatchGuard swg(swReunitize);
	lattice_apply([](auto &a) { a = project_on_group_fast(a); }, U);
}

template <typename vG> void reunitize(GaugeField<vG> &U)
{
	util::StopwatchGuard swg(swReunitize);
	lattice_apply([](auto &a) { a = project_on_group_fast(a); }, U);
}

/** normalized to [0,1] */
template <typename G> double plaquette(GaugeField<G> const &U)
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
			s += sum_real_trace(tmp);
		}
	return s / (vol * Nd * (Nd - 1) * 0.5) / GaugeTraits<G>::Nc();
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

template <typename G>
Lattice<G> wilsonDeriv(GaugeField<G> const &U, int mu, double beta)
{
	return project_on_algebra(U[mu] * stapleSum(U, mu) *
	                          (beta / (2 * GaugeTraits<G>::Nc())));
}

// calls f<vG>() with vG determined at runtime based on gauge group and FP prec
template <typename F>
void dispatchByGroup(F f, std::string const &group, int precision)
{
	if (precision == 1)
	{
		if (group == "u1")
			f.template operator()<U1<float>>();
		else if (group == "su2")
			f.template operator()<SU2<float>>();
		else if (group == "su3")
			f.template operator()<SU3<float>>();
		else
			throw std::runtime_error(
			    fmt::format("unknown gauge group '{}'", group));
	}
	else if (precision == 2)
	{
		if (group == "u1")
			f.template operator()<U1<double>>();
		else if (group == "su2")
			f.template operator()<SU2<double>>();
		else if (group == "su3")
			f.template operator()<SU3<double>>();
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
	assert(false && "readConfig() not implemented right now");
	(void)configName;
	(void)expected_grid;
	// "my_ensemble.h5/configs/100"
	/*if (auto p = configName.find(".h5/"); p != std::string::npos)
	{
	    auto filename = configName.substr(0, p + 3);
	    auto dset = configName.substr(p + 3);
	    auto file = util::Hdf5File::open(filename);
	    auto geom = file.get_attribute<std::vector<int>>("geometry");
	    auto grid = Grid(Coordinate(geom.begin(), geom.end()));
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
	        fmt::format("unknown config file format '{}'", configName));*/
}

} // namespace mesh
