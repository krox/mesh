#include "gauge/markov.h"

#include "groups/su2.h"
#include "groups/su3.h"
#include "groups/u1.h"
#include "groups/z2.h"

template <typename G>
GaugeChainResult runChainImpl(const GaugeChainParams &param,
                              const WilsonActionParams &actionParams)
{
	/** initialize field */
	GaugeMesh<G> mesh(std::make_shared<GaugeTopology>(param.geom));
	WilsonAction<G> action(mesh, actionParams, param.seed);

	/** run the Markov chain */
	GaugeChainResult res;
	res.plaqHistory = xt::zeros<double>({param.count});
	res.topHistory = xt::zeros<double>({param.count});

	mesh.initUnit();
	for (int i = -param.discard; i < param.count; ++i)
	{
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
		res.topHistory(i) = mesh.topCharge(); // TODO: smearing
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
