#include "gauge/gauge.h"

#include "gauge/wilson.h"
#include "groups/su2.h"
#include "groups/u1.h"
#include "groups/z2.h"
#include "xtensor/xview.hpp"
#include <fmt/format.h>

template <typename Action>
gauge_chain_result_t runChain(const gauge_chain_param_t<Action> &param)
{
	/** initialize field */
	gauge_mesh<typename Action::G> mesh(param.geom);
	Action action(mesh, param.param, param.seed);

	/** run the Markov chain */
	gauge_chain_result_t res;
	res.plaqHistory = xt::zeros<double>({param.count});

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
		res.plaqHistory(i) = action.action(); // action density
	}

	return res;
}

template gauge_chain_result_t
runChain<wilson_action<Z2>>(const gauge_chain_param_t<wilson_action<Z2>> &);
template gauge_chain_result_t
runChain<wilson_action<U1>>(const gauge_chain_param_t<wilson_action<U1>> &);
template gauge_chain_result_t
runChain<wilson_action<SU2>>(const gauge_chain_param_t<wilson_action<SU2>> &);
