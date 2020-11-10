#include "gauge/markov.h"

#include "fmt/format.h"

#include "groups/su2.h"
#include "groups/su3.h"
#include "groups/u1.h"
#include "groups/z2.h"
#include "util/hdf5.h"

template <typename G>
GaugeChainResult runChainImpl(const GaugeChainParams &param,
                              const WilsonActionParams &actionParams)
{
	/** initialize field */
	GaugeMesh<G> mesh(param.top);
	WilsonAction<G> action(mesh, actionParams, param.seed);

	/** run the Markov chain */
	GaugeChainResult res;
	res.plaqHistory = xt::zeros<double>({param.count});
	res.topHistory = xt::zeros<double>({param.count});

	DataFile file;
	if (param.filename != "")
	{
		file = DataFile::create(param.filename);

		// physical parameters
		file.setAttribute("group", param.group);
		file.setAttribute("beta", actionParams.beta);
		file.setAttribute("c0", actionParams.c0);
		file.setAttribute("c1", actionParams.c1);
		file.setAttribute("geometry", param.top->geom);

		// simulation parameters
		file.setAttribute("markov_count", param.count);
		file.setAttribute("markov_discard", param.discard);
		file.setAttribute("markov_sweeps", param.sweeps);

		file.makeGroup("/configs");
	}

	mesh.initUnit();
	for (int i = -param.discard; i < param.count; ++i)
	{
		if (param.filename != "")
			fmt::print("Markov step {}\n", i + 1);

		for (int j = 0; j < param.sweeps; ++j)
		{
			action.sweep();
			for (int k = 0; k < param.clusters; ++k)
				action.cluster();
		}
		if (i < 0)
			continue;

		// basic observables
		res.plaqHistory(i) = mesh.plaqAvg();
		// res.topHistory(i) = mesh.topCharge(); // TODO: smearing

		if (param.filename != "")
		{
			std::string name = fmt::format("/configs/{}", i + 1);
			file.createData(name, {(unsigned)mesh.nLinks(), G::repSize()})
			    .write(mesh.rawConfig());
		}
	}

	if (param.filename != "")
	{
		file.createData("plaq_history", {res.plaqHistory.size()})
		    .write(res.plaqHistory);
	}

	return res;
}

GaugeChainResult runChain(const GaugeChainParams &chainParams,
                          const WilsonActionParams &actionParams)
{
	if (chainParams.group == "z2")
		return runChainImpl<Z2>(chainParams, actionParams);
	if (chainParams.group == "u1")
		return runChainImpl<U1>(chainParams, actionParams);
	if (chainParams.group == "su2")
		return runChainImpl<SU2>(chainParams, actionParams);
	if (chainParams.group == "su3")
		return runChainImpl<SU3>(chainParams, actionParams);
	assert(false);
}
