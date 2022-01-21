#pragma once

#include "lattice/gauge.h"
#include "util/gnuplot.h"
#include "util/linalg.h"
#include "util/progressbar.h"
#include "util/random.h"
#include "util/simd.h"
#include <cmath>
#include <string_view>
#include <vector>

namespace mesh {

using std::exp;

// create an array of delta values for a single Hamiltonian dynamics trajectory
//   * epsilon is in Langevin-units, result is in HMD units
//   * resulting deltas should be interpreted alternating between
//     position/momentum steps
//   * supported schemes are: 2lf, 2mn, 4mn
std::vector<double> makeDeltas(std::string_view scheme, double epsilon,
                               int substeps);

// run a single HMD trajectory with basic Wilson-action
//   * should be generalized to other actions
template <typename vG>
void runHmd(GaugeField<vG> &U, GaugeField<vG> &P, double beta,
            std::vector<double> const &deltas)
{
	assert(deltas.size() % 2 == 1);
	assert(U.size() == P.size() && !U.empty());
	int Nd = U[0].grid().ndim();

	for (size_t i = 0; i < deltas.size(); ++i)
		if (i % 2 == 0)
		{
			for (int mu = 0; mu < Nd; ++mu)
				U[mu] = exp(P[mu] * deltas[i]) * U[mu];
		}
		else
		{
			for (int mu = 0; mu < Nd; ++mu)
				P[mu] -= wilsonDeriv(U, mu, beta) * deltas[i];
		}
}

struct HmcParams
{
	// physics parameters
	std::string group = "su3";
	std::vector<int32_t> geom = {6, 6, 6, 6};
	double beta = 6.0;

	// simulation parameters
	std::string scheme = "4mn";
	double epsilon = 1.0;
	int substeps = 8;
	int count = 100;
	int seed = -1;
	int precision = 2; // 1=float, 2=double

	// others
	bool doPlot = false;
};

// explicitly instantiated for all reasonable combinations of
// gauge group, simd-width, floating point precision
template <typename vG> void runHmc_impl(HmcParams const &params);

inline void runHmc(HmcParams const &params)
{
	if (params.precision == 1)
	{
		if (params.group == "u1")
			runHmc_impl<U1<util::simd<float>>>(params);
		else if (params.group == "su2")
			runHmc_impl<SU2<util::simd<float>>>(params);
		else if (params.group == "su3")
			runHmc_impl<SU3<util::simd<float>>>(params);
		else
			throw std::runtime_error(
			    fmt::format("unknown gauge group '{}'", params.group));
	}
	else if (params.precision == 2)
	{
		if (params.group == "u1")
			runHmc_impl<U1<util::simd<double>>>(params);
		else if (params.group == "su2")
			runHmc_impl<SU2<util::simd<double>>>(params);
		else if (params.group == "su3")
			runHmc_impl<SU3<util::simd<double>>>(params);
		else
			throw std::runtime_error(
			    fmt::format("unknown gauge group '{}'", params.group));
	}
	else
		throw std::runtime_error(
		    fmt::format("invlid precision level '{}'", params.precision));
}

} // namespace mesh
