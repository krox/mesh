#include "scalar/scalar.h"

#include "scalar/phi4.h"

#include "util/fft.h"

template <typename Action>
scalar_chain_result_t runChain(const scalar_chain_param_t<Action> &param)
{
	/** initialize field */
	scalar_mesh<1> mesh(Topology::lattice(param.geom));
	Action action(mesh, param.param, param.seed);
	Correlator corr(mesh.phi.data(), param.geom);

	/** run the Markov chain */
	scalar_chain_result_t res;
	res.c2pt = xt::zeros<double>({param.count, mesh.nSites()});
	mesh.initZero();
	for (int i = -param.discard; i < param.count; ++i)
	{
		for (int j = 0; j < param.sweeps; ++j)
			action.sweep();
		if (i < 0)
			continue;

		// write 2pt correlator
		corr.compute();
		xt::view(res.c2pt, i) = corr();
	}

	/** analyze measurements */
	res.reject = (double)action.nReject / (action.nAccept + action.nReject);

	return res;
}

template scalar_chain_result_t
runChain<phi4_action>(const scalar_chain_param_t<phi4_action> &);
