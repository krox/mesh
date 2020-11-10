#ifndef SCALAR_SCALAR_H
#define SCALAR_SCALAR_H

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

	std::vector<std::vector<int>> g;
	std::vector<Scalar> phi;

	int nFlavors() const { return N; }
	int nSites() const { return (int)g.size(); }
	int nLinks() const
	{
		int count = 0;
		for (int i = 0; i < nSites(); ++i)
			count += (int)g[i].size();
		return count / 2;
	}

	/** does not initialize the field */
	ScalarMesh(const Topology &top) : g(top.nSites()), phi(top.nSites())
	{
		for (auto &[a, b] : top.links)
		{
			g[a].push_back(b);
			g[b].push_back(a);
		}
	}

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
template <typename Action> struct scalar_chain_param_t
{

	/** parameters with physical meaning */
	std::vector<int> geom = {128};  // size of lattice
	typename Action::param_t param; // parameters of action

	/** additional simulation parameters */
	int count = 100;   // number of configs to generate
	int discard = 0;   // number of discarded configs
	int sweeps = 1;    // number of HB-sweeps between measurements
	int clusters = 0;  // number of cluster-flips per HB sweeps
	uint64_t seed = 0; // seed for random number generator
	std::string filename = "";
};

/** some measurements taken during the simulation. This may include measurements
 * on intermediate configs that were not saved. */
struct scalar_chain_result_t
{
	double reject;

	std::vector<double> actionHistory;
	std::vector<double> magHistory;
};

template <typename Action>
scalar_chain_result_t runChain(const scalar_chain_param_t<Action> &param);

#endif
