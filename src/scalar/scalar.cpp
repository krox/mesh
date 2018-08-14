#include "scalar/scalar.h"

#include <cassert>
#include <fmt/format.h>
#include <random>
#include <vector>

#include "mesh/topology.h"
#include "util/fft.h"
#include "util/gnuplot.h"
#include "util/io.h"
#include "util/random.h"

namespace scalar {

struct ScalarField
{
	Topology top;
	std::vector<double> phi;

	ScalarField(Topology top) : top(top), phi(top.nSites(), 0.0) {}

	/** one heat-bath sweep */
	void sweep(rng_t &rng, double kappa)
	{
		std::normal_distribution dist(0.0, 1.0 / sqrt(2.0));
		for (int i = 0; i < top.nSites(); ++i)
		{
			double rho = 0;
			for (auto [j, _] : top.graph[i])
				rho += phi[j];
			phi[i] = dist(rng) + rho * (kappa / top.graph[i].size());
		}
	}

	/** 2pt correlator (separation=0) */
	double c2pt0()
	{
		double sum = 0.0;
		for (int i = 0; i < top.nSites(); ++i)
			sum += phi[i] * phi[i];
		return sum / top.nSites();
	}
};

std::string ChainParams::autoFilename() const
{
	char g = 's';

	return fmt::format("{}{}.k{}.l{}.h5", g, geom[0], (int)(kappa * 1000),
	                   (int)(lambda * 1000));
}

ChainResult runChain(const ChainParams &params)
{
	rng_t rng(params.seed);

	/** initialize field */
	assert(params.geom.size() == 1);
	auto field = ScalarField(Topology::lattice(params.geom));
	Correlator corr(params.geom[0]);

	DataFile file;
	DataSet c2pt;
	if (params.filename != "")
	{
		file = DataFile::create(params.filename);

		// physical parameters
		file.setAttribute("kappa", params.kappa);
		file.setAttribute("lambda", params.lambda);

		// topological parameters
		file.setAttribute("topology", "periodic1");
		file.setAttribute("geometry", params.geom);

		// simulation parameters
		file.setAttribute("markov_count", params.count);
		file.setAttribute("markov_discard", params.discard);
		file.setAttribute("markov_sweeps", params.sweeps);
		file.setAttribute("markov_overrelax", params.overrelax);

		file.makeGroup("/configs");

		c2pt = file.createData(
		    "c2pt", {(hsize_t)params.count, (hsize_t)params.geom[0]});
	}

	/** run the Markov chain */
	ChainResult res;
	for (int i = -params.discard; i < params.count; ++i)
	{
		// do some thermalization sweeps and basic measurements
		for (int j = 0; j < params.sweeps; ++j)
		{
			// one heat-bath sweep
			field.sweep(rng, params.kappa);

			// multiple OR sweeps
			/*for (int k = 0; k < params.overrelax; ++k)
			   field.sweepOR(params.kappa);*/

			if (i >= 0)
				res.actionHistory.push_back(field.c2pt0());
		}
		if (i < 0)
			continue;

		if (params.filename != "")
		{
			// write config
			std::string name = fmt::format("/configs/{}", i + 1);
			file.createData(name, {(unsigned)field.top.nSites()})
			    .write(field.phi);

			// write 2pt correlator
			corr.compute(field.phi);
			c2pt.write(i, corr());
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

} // namespace scalar
