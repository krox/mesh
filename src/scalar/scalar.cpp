#include "scalar/scalar.h"

#include "scalar/ising.h"
#include "scalar/phi4.h"
#include "scalar/sigma.h"
#include "util/fft.h"
#include "xtensor/xview.hpp"
#include <fmt/format.h>

template <typename Action>
scalar_chain_result_t runChain(const scalar_chain_param_t<Action> &param)
{
	/** initialize field */
	scalar_mesh<Action::rep> mesh(Topology::lattice(param.geom));
	Action action(mesh, param.param, param.seed);
	Correlator corr(mesh.phi.data(), param.geom);

	/** run the Markov chain */
	scalar_chain_result_t res;
	std::vector<hsize_t> s;
	s.push_back(param.count);
	for (int d : param.geom)
		s.push_back(d);
	res.c2pt = xt::zeros<double>(s);
	res.actionHistory = xt::zeros<double>({param.count});
	res.magHistory = xt::zeros<double>({param.count});
	res.phaseAngle = xt::zeros<double>({param.count});

	DataFile file;
	if (param.filename != "")
	{
		file = DataFile::create(param.filename);

		// physical parameters
		file.setAttribute("beta", param.param.beta);
		file.setAttribute("mu", param.param.mu);

		// topological parameters
		file.setAttribute("geometry", param.geom);

		// simulation parameters
		file.setAttribute("markov_count", param.count);
		file.setAttribute("markov_discard", param.discard);
		file.setAttribute("markov_sweeps", param.sweeps);

		if (!param.skipConfig)
			file.makeGroup("/configs");
	}

	mesh.initZero();
	for (int i = -param.discard; i < param.count; ++i)
	{
		for (int j = 0; j < param.sweeps; ++j)
			action.sweep();
		if (i < 0)
			continue;

		// basic observables
		res.actionHistory(i) = action.action(); // action density(real part)
		res.magHistory(i) = action.magnetization();
		res.phaseAngle(i) = action.phaseAngle(); // imaginary part of action

		// measure 2pt correlator
		corr.compute();
		xt::view(res.c2pt, i) = corr.fullCorr();

		// write config to file
		if (param.filename != "")
		{
			std::vector<hsize_t> shape;
			for (int d : param.geom)
				shape.push_back(d);
			shape.push_back(Action::rep);

			if (!param.skipConfig)
			{
				std::string name = fmt::format("/configs/{}", i + 1);
				file.createData(name, shape).write(mesh.rawConfig());
			}
		}
	}

	/** analyze measurements */
	res.reject = (double)action.nReject / (action.nAccept + action.nReject);

	if (param.filename != "")
	{
		file.createData("action_history", {res.actionHistory.size()})
		    .write(res.actionHistory);
		file.createData("mag_history", {res.magHistory.size()})
		    .write(res.magHistory);
		file.createData("phase_angle", {res.actionHistory.size()})
		    .write(res.phaseAngle);
		file.createData("c2pt", s).write(res.c2pt);
	}

	return res;
}

template scalar_chain_result_t
runChain<phi4_action>(const scalar_chain_param_t<phi4_action> &);
template scalar_chain_result_t
runChain<sigma_action>(const scalar_chain_param_t<sigma_action> &);
template scalar_chain_result_t
runChain<ising_action>(const scalar_chain_param_t<ising_action> &);
