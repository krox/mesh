#include "mesh/markov.h"

#include <fmt/format.h>

#include "mesh/gauge_action.h"
#include "mesh/mesh.h"
#include "mesh/su2.h"
#include "mesh/u1.h"
#include "mesh/z2.h"
#include "util/io.h"
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

	DataFile file;
	if (params.filename != "")
	{
		file = DataFile::create(params.filename);

		// physical parameters
		file.setAttribute("group", params.group);
		file.setAttribute("beta", params.beta);
		file.setAttribute("beta2", params.beta2);

		// topological parameters
		file.setAttribute("topology", "periodic4");
		std::vector<int> geom = {params.n, params.n, params.n, params.n};
		file.setAttribute("geometry", geom);

		// simulation parameters
		file.setAttribute("markov_count", params.count);
		file.setAttribute("markov_discard", params.discard);
		file.setAttribute("markov_sweeps", params.sweeps);

		file.makeGroup("/configs");
	}

	/** run the Markov chain */
	ChainResult res;
	for (int i = -params.discard; i < params.count; ++i)
	{
		// do some thermalization sweeps and basic measurements
		for (int j = 0; j < params.sweeps; ++j)
		{
			ga.thermalize(rng, params.beta, params.beta2);
			if (i >= 0)
				res.actionHistory.push_back(ga.loop4());
		}
		if (i < 0)
			continue;

		if (params.filename != "")
		{
			std::string name = fmt::format("/configs/{}", i + 1);
			file.createData(name, {(unsigned)m.top.nLinks(), G::repSize()})
			    .write(m.rawLinksConst());
		}
	}

	/** analyze measurements */
	res.action = mean(res.actionHistory);
	res.corrTime = correlationTime(res.actionHistory) / params.sweeps;

	if (params.filename != "")
	{
		file.createData("action_history", {res.actionHistory.size()})
		    .write(res.actionHistory);
	}

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
