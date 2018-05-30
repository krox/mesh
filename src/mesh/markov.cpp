#include "mesh/markov.h"

#include <fmt/format.h>

#include "mesh/gauge_action.h"
#include "mesh/mesh.h"
#include "mesh/su2.h"
#include "mesh/u1.h"
#include "mesh/z2.h"
#include "util/stats.h"

template <typename G> ChainResult runChainImpl(const ChainParams &params)
{
	rng_t rng(params.seed);

	/** initialize mesh */
	auto m = Mesh<G>(Topology::lattice4D(params.n));
	if (params.init == 1)
		m.initOrdered();
	else if (params.init == 2)
		m.initRandom(rng);
	else if (params.init == 3)
		m.initMixed(rng);
	else
		throw std::runtime_error("unknown initialization");
	auto ga = GaugeAction(m);

	/** run the Markov chain */
	ChainResult res;
	for (int i = -params.nWarms; i < params.count * params.nSweeps; ++i)
	{
		// do a thermalization sweep and basic measurements
		ga.thermalize(rng, params.beta, params.beta2);
		res.actionHistory.push_back(ga.loop4());

		if (i >= 0 && i % params.nSweeps == 0)
		{
			/** TODO: write config to HDF5 file */
		}
	}

	/** analyze measurements */
	Autocorrelation ac;
	for (int i = 0; i < params.count * params.nSweeps; ++i)
		ac.add(res.actionHistory.at(params.nWarms + i));
	res.corrTime = ac.corrTime() / params.nSweeps;
	res.action = ac.mean();

	return res;
}

ChainResult runChain(const ChainParams &params)
{
	if (params.group == "z2")
		return runChainImpl<Z2>(params);
	else if (params.group == "u1")
		return runChainImpl<U1>(params);
	else if (params.group == "su2")
		return runChainImpl<SU2>(params);
	else
		throw std::runtime_error(
		    fmt::format("unknown gauge group '{}'", params.group));
}
