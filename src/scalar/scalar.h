#ifndef SCALAR_SCALAR_H
#define SCALAR_SCALAR_H

#include <cstdint>
#include <string>
#include <vector>

#include "xtensor/xarray.hpp"

#include "groups/scalar.h"
#include "mesh/topology.h"
#include "util/io.h"
#include "util/span.h"

template <int N> class scalar_mesh {
  public:
	std::vector<std::vector<int>> g;
	std::vector<int> timeStep;
	std::vector<Scalar<N>> phi;

	int nFlavors() const { return N; }
	int nSites() const { return (int)g.size(); }
	int nLinks() const {
		int count = 0;
		for (int i = 0; i < nSites(); ++i)
			count += (int)g[i].size();
		return count / 2;
	}

	/** does not initialize the field */
	scalar_mesh(const Topology &top)
	    : g(top.nSites()), timeStep(top.timeStep), phi(top.nSites()) {
		for (auto &[a, b] : top.links) {
			g[a].push_back(b);
			g[b].push_back(a);
		}
	}

	void initZero() {
		for (auto &x : phi)
			x = Scalar<N>::zero();
	}

	void initOne() {
		for (auto &x : phi)
			x = Scalar<N>::one();
	}

	span<const double> rawConfig() const {
		return span<const double>((double const *)phi.data(), phi.size() * N);
	}
};

/** parameters of markov chain */
template <typename Action> struct scalar_chain_param_t {

	/** parameters with physical meaning */
	std::vector<int> geom = {128};  // size of lattice
	typename Action::param_t param; // parameters of action

	/** additional simulation parameters */
	int count = 100;   // number of configs to generate
	int discard = 0;   // number of discarded configs
	int sweeps = 1;    // number of sweeps between measurements
	uint64_t seed = 0; // seed for random number generator
	std::string filename = "";
};

/** some measurements taken during the simulation. This may include measurements
 * on intermediate configs that were not saved. */
struct scalar_chain_result_t {
	double reject;

	xt::xarray<double> c2pt;
	xt::xarray<double> actionHistory;
	xt::xarray<double> phaseAngle;
};

template <typename Action>
scalar_chain_result_t runChain(const scalar_chain_param_t<Action> &param);

#endif
