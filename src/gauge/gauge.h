#ifndef GAUGE_GAUGE_H
#define GAUGE_GAUGE_H

#include <cstdint>
#include <string>
#include <vector>

#include "xtensor/xarray.hpp"

#include "mesh/topology.h"
#include "util/io.h"
#include "util/span.h"

template <typename G> class gauge_mesh
{
  public:
	std::vector<int> geom;
	std::vector<G> u;

	int nSites() const
	{
		int s = 1;
		for (int n : geom)
			s *= n;
		return s;
	}

	int nLinks() const { return (int)u.size(); }

	/** initializes to unit-field */
	gauge_mesh(const std::vector<int> &geom)
	    : geom(geom), u(nSites() * geom.size(), G::one())
	{}

	void initUnit()
	{
		for (auto &x : u)
			x = G::one();
	}

	span<const double> rawConfig() const
	{
		return span<const double>((double const *)u.data(),
		                          u.size() * G::repSize());
	}
};

/** parameters of markov chain */
template <typename Action> struct gauge_chain_param_t
{
	/** parameters with physical meaning */
	std::vector<int> geom = {4, 4, 4, 4}; // size of lattice
	typename Action::param_t param;       // parameters of action

	/** additional simulation parameters */
	int count = 100;   // number of configs to generate
	int discard = 0;   // number of discarded configs
	int sweeps = 1;    // number of HB-sweeps between measurements
	int clusters = 0;  // number of cluster-flips per HB sweeps
	uint64_t seed = 0; // seed for random number generator
	std::string filename = "";
	bool skipConfig = false;
};

/** some measurements taken during the simulation. This may include measurements
 * on intermediate configs that were not saved. */
struct gauge_chain_result_t
{
	xt::xarray<double> plaqHistory; // average plaquette
	// xt::xarray<double> topHistory;  // global topological charge
};

template <typename Action>
gauge_chain_result_t runChain(const gauge_chain_param_t<Action> &param);

#endif
