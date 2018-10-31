#include "scalar/scalar.h"

#include "scalar/phi4.h"
#include "scalar/sigma.h"
#include "util/fft.h"
#include "xtensor/xview.hpp"

template <typename Action>
scalar_chain_result_t runChain(const scalar_chain_param_t<Action> &param)
{
	/** initialize field */
	scalar_mesh<Action::rep> mesh(Topology::lattice(param.geom));
	Action action(mesh, param.param, param.seed);
	Correlator corr(mesh.phi.data(), param.geom);

	/** run the Markov chain */
	scalar_chain_result_t res;
	std::vector<int> s;
	s.push_back(param.count);
	for (int d : param.geom)
		s.push_back(d);
	res.c2pt = xt::zeros<double>(s);
	res.actionHistory = xt::zeros<double>({param.count});
	res.phaseAngle = xt::zeros<double>({param.count});
	mesh.initZero();
	for (int i = -param.discard; i < param.count; ++i)
	{
		for (int j = 0; j < param.sweeps; ++j)
			action.sweep();
		if (i < 0)
			continue;

		// basic observables
		res.actionHistory(i) = action.action();  // action density(real part)
		res.phaseAngle(i) = action.phaseAngle(); // imaginary part of action

		// measure 2pt correlator
		corr.compute();
		xt::view(res.c2pt, i) = corr.fullCorr();
	}

	/** analyze measurements */
	res.reject = (double)action.nReject / (action.nAccept + action.nReject);

	return res;
}

template scalar_chain_result_t
runChain<phi4_action>(const scalar_chain_param_t<phi4_action> &);
template scalar_chain_result_t
runChain<sigma_action>(const scalar_chain_param_t<sigma_action> &);
