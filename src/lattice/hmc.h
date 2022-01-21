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
				U[mu] = exp(P[mu] * deltas[i]) * U[mu];
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
template <typename vG> struct Hmc
{
	// state of the simulation
	Grid const &g;        // lattice grid
	GaugeField<vG> U;     // gauge field
	GaugeField<vG> P;     // conjugate momenta
	util::xoshiro256 rng; // random number generator

	// observables (reset any time using .clear_observables())
	std::vector<double> plaq_history;
	std::vector<double> deltaH_history, accept_history;

	GaugeField<vG> U_new; // temporary

	Hmc(Grid const &g)
	    : g(g), U(makeGaugeField<vG>(g)), P(makeGaugeField<vG>(g))
	{}

	// reset the gauge field to a random config
	void randomizeGaugeField() { randomGaugeField(U, rng); }

	// new gaussian momenta
	void randomizeMomenta()
	{
		// NOTE on conventions:
		//     * H = S(U) + 1/2 P^i P^i = S(U) - tr(P*P) = S(U) + norm2(P)
		//     * U' = P, P' = -S'(U)
		randomAlgebraField(P, rng);
	}

	// generate momenta -> run a trajectory -> accept/reject it -> measure
	// (NOTE: even if rejected, old momenta are destroyed)
	void runHmcUpdate(double beta, std::vector<double> const &deltas)
	{
		// generate new momenta
		randomizeMomenta();

		// make proposal
		double H_old = wilsonAction(U, beta) + norm2(P);
		U_new = U;
		runHmd(U_new, P, beta, deltas);
		reunitize(U_new); // no idea if this is the best place to put it
		double H_new = wilsonAction(U_new, beta) + norm2(P);
		auto deltaH = H_new - H_old;

		// metropolis step
		if (rng.uniform() < exp(-deltaH))
		{
			accept_history.push_back(1.0);
			std::swap(U, U_new);
		}
		else
			accept_history.push_back(0.0);

		// track some observables
		plaq_history.push_back(plaquette(U));
		deltaH_history.push_back(deltaH);
	}
};

} // namespace mesh
