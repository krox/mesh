#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mesh/topology.h"
#include "util/hdf5.h"
#include "util/linalg.h"
#include "util/random.h"
#include "util/span.h"

using rng_t = util::xoshiro256;

/**
 * ScalarMesh = Topology + Field values.
 *   - Does not know the used action, but only the size of the representation.
 */
template <int N> class ScalarMesh
{
  public:
	using Scalar = util::Vector<double, N>;

	Topology top;
	std::vector<Scalar> phi;

	int nFlavors() const { return N; }
	int nSites() const { return top.nSites(); }
	int nLinks() const { return top.nLinks(); }

	/** does not initialize the field */
	ScalarMesh(const Topology &top_) : top(top_), phi(top_.nSites()) {}

	void initZero()
	{
		for (auto &x : phi)
			for (size_t i = 0; i < N; ++i)
				x[i] = 0.0;
	}

	void initOne()
	{
		for (auto &x : phi)
			for (size_t i = 0; i < N; ++i)
				x[i] = i == 0 ? 1.0 : 0.0;
	}

	util::span<const double> rawConfig() const
	{
		return util::span<const double>((double const *)phi.data(),
		                                phi.size() * N);
	}
};

/** parameters of markov chain */
template <typename Action> struct ScalarChainParams
{
	// physical parameters
	std::vector<int> geom = {128};          // size of lattice
	typename Action::params_t actionParams; // parameters of action

	// additional parameters for the Markov chain
	int count = 100;   // number of configs to generate
	int discard = 0;   // number of discarded configs
	int sweeps = 1;    // number of HB-sweeps between measurements
	int clusters = 0;  // number of cluster-flips per HB sweeps
	uint64_t seed = 0; // seed for random number generator

	// output of configs
	std::string filename = "";
};

/** some measurements taken during the simulation. This may include measurements
 * on intermediate configs that were not saved. */
struct ScalarChainResult
{
	double reject;

	std::vector<double> actionHistory;
	std::vector<double> magHistory;
};

template <typename Action>
ScalarChainResult runChain(const ScalarChainParams<Action> &params);
