#include "mesh/markov.h"

#include <fmt/format.h>

#include "mesh/gauge_action.h"
#include "mesh/mesh.h"
#include "mesh/su2.h"
#include "mesh/u1.h"
#include "mesh/z2.h"
#include "util/io.h"
#include "util/stats.h"

std::string ChainParams::autoFilename() const
{
	char g = 'x';
	if (geom.size() == 4 && geom[0] == geom[1] && geom[1] == geom[2])
	{
		if (geom[3] == geom[0])
			g = 'p';
		if (geom[3] == 2 * geom[0])
			g = 'q';
		if (geom[3] == 3 * geom[0])
			g = 'r';
	}

	if (beta2 == 0)
		return fmt::format("{}.{}{}.b{}.h5", group, g, geom[0],
		                   (int)(beta * 1000));
	else
		return fmt::format("{}.{}{}.b{}.b{}.h5", group, g, geom[0],
		                   (int)(beta * 1000), (int)(beta2 * 1000));
}

template <typename G> ChainResult runChainImpl(const ChainParams &params)
{
	rng_t rng(params.seed);

	/** initialize mesh */
	assert(params.geom.size() == 4);
	auto m = Mesh<G>(Topology::lattice(params.geom));
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
		file.setAttribute("geometry", params.geom);

		// simulation parameters
		file.setAttribute("markov_count", params.count);
		file.setAttribute("markov_discard", params.discard);
		file.setAttribute("markov_sweeps", params.sweeps);
		file.setAttribute("markov_overrelax", params.overrelax);

		file.makeGroup("/configs");
	}

	/** run the Markov chain */
	ChainResult res;
	for (int i = -params.discard; i < params.count; ++i)
	{
		// do some thermalization sweeps and basic measurements
		for (int j = 0; j < params.sweeps; ++j)
		{
			// one heat-bath sweep
			ga.thermalize(rng, params.beta, params.beta2);

			// multiple OR sweeps
			for (int k = 0; k < params.overrelax; ++k)
				ga.overrelax();

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
