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

inline util::Stopwatch swExp;

// create an array of delta values for a single Hamiltonian dynamics trajectory
//   * epsilon is in Langevin-units, result is in HMD units
//   * resulting deltas should be interpreted alternating between
//     position/momentum steps
//   * supported schemes are: 2lf, 2mn, 4mn
std::vector<double> makeDeltas(std::string_view scheme, double epsilon,
                               int substeps);

// same es 'U = exp(P * t) * U', but slightly faster (and no allocations)
template <typename vG>
void evolve(Lattice<vG> &U, Lattice<vG> const &P, double t)
{
	util::StopwatchGuard swg(swExp);
	assert(compatible(U, P));
	for (size_t i = 0; i < U.grid().osize(); ++i)
		U.data()[i] = exp(P.data()[i] * t) * U.data()[i];
}

// evolve (U,P) in Hamiltonian dynamics basic Wilson-action
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
				evolve(U[mu], P[mu], deltas[i]);
		}
		else
		{
			for (int mu = 0; mu < Nd; ++mu)
				P[mu] -= wilsonDeriv(U, mu, beta) * deltas[i];
		}
}

// This holds the state of the simulation and some tracked statistics.
// Does not neccessarily hold all the parameters which might change a lot during
// a complicated simulation setup.
//     * 'beta' should be generalized to an arbitrary action
//     * 'deltas' should be generalized to an arbitrary integration scheme
template <typename vG> class Hmc
{
  public:
	// state of the simulation
	Grid const &g;        // lattice grid
	GaugeField<vG> U;     // gauge field
	GaugeField<vG> P;     // conjugate momenta
	util::xoshiro256 rng; // random number generator

	// observables (reset any time using .reset_observables())
	std::vector<double> plaq_history;
	std::vector<double> deltaH_history, accept_history;

	void reset_observables()
	{
		plaq_history.clear();
		deltaH_history.clear();
		accept_history.clear();
	}

	GaugeField<vG> U_new; // temporary

	Hmc(Grid const &g);

	// reset the gauge field to a random config
	void randomizeGaugeField();

	// new gaussian momenta
	void randomizeMomenta();

	// generate momenta -> run a trajectory -> accept/reject it -> measure
	// (NOTE: even if rejected, old momenta are destroyed)
	void runHmcUpdate(double beta, std::vector<double> const &deltas);
};

} // namespace mesh
