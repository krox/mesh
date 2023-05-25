#pragma once

#include "gauge/utils.h"
#include "util/gnuplot.h"
#include "util/linalg.h"
#include "util/progressbar.h"
#include "util/random.h"
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

// evolution of gauge field:
// U = exp(project_algebra(P) * t) * U
template <typename G> void evolve(Lattice<G> &U, Lattice<G> const &P, double t)
{
	util::StopwatchGuard swg(swExp);
	lattice_apply(
	    [t](auto &u, auto const &p) {
		    u = gauge::exp(project_on_algebra(p) * t) * u;
	    },
	    U, P);
}

// evolve (U,P) in Hamiltonian dynamics basic Wilson-action
//   * should be generalized to other actions
template <typename G>
void runHmd(GaugeField<G> &U, GaugeField<G> &P, double beta,
            std::vector<double> const &deltas)
{
	assert(deltas.size() % 2 == 1);
	assert(U.grid() == P.grid());
	int Nd = U.grid().ndim();

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
template <typename G> class Hmc
{
  public:
	// state of the simulation
	Grid const &g;        // lattice grid
	GaugeField<G> U;      // gauge field
	GaugeField<G> P;      // conjugate momenta
	util::xoshiro256 rng; // random number generator

	// observables (reset any time using .reset_observables())
	std::vector<double> plaq_history;
	std::vector<double> deltaH_history, accept_history;
	std::vector<double> time_history;

	void reset_observables()
	{
		plaq_history.clear();
		deltaH_history.clear();
		accept_history.clear();
		time_history.clear();
	}

	void print_summary()
	{
		using util::mean, util::variance, util::min, util::max;

		fmt::print("========== HMC summary ==========\n");
		fmt::print("time per step = {:.3f} s (min = {:.3f}, max = {:.3f})\n",
		           mean(time_history), min(time_history), max(time_history));
		fmt::print("plaquette = {:.4f} +- {:.4f}\n", mean(plaq_history),
		           sqrt(variance(plaq_history) / plaq_history.size()));
		fmt::print("acceptance = {:.2f}\n", mean(accept_history));
		fmt::print("<exp(-dH)> = {:.4f}\n",
		           mean(deltaH_history, [](double x) { return exp(-x); }));
	}

	GaugeField<G> U_new; // temporary

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
