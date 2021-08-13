#include "scalar/scalar.h"

#include "scalar/ising.h"
#include "scalar/phi4.h"
#include "scalar/sigma.h"
#include "util/progressbar.h"
#include <fmt/format.h>

template <typename Action>
ScalarChainResult runChain(const ScalarChainParams<Action> &params)
{
	assert(params.hdf5Type > 0);

	typename Action::mesh_t mesh(Topology::lattice(params.geom));
	Action action(params.actionParams);
	rng_t rng(params.seed);
	ScalarChainResult res;

	util::DataFile file;

	if (params.filename != "")
	{
		file =
		    util::DataFile::create(params.filename, params.overwrite_existing);

		// physical parameters
		file.setAttribute("beta", params.actionParams.beta);
		// file.setAttribute("mu", params.actionParams.mu);

		// topological parameters
		file.setAttribute("geometry", params.geom);

		// simulation parameters
		file.setAttribute("markov_count", params.count);
		file.setAttribute("markov_discard", params.discard);
		file.setAttribute("markov_sweeps", params.sweeps);
		file.setAttribute("markov_clusters", params.clusters);

		file.makeGroup("/configs");
	}

	mesh.initZero();
	auto pb = util::ProgressBar(params.count + params.discard);
	for (int i = -params.discard; i < params.count; ++i, ++pb)
	{
		pb.show();
		for (int j = 0; j < params.sweeps; ++j)
		{
			action.sweep(mesh, rng);
			for (int k = 0; k < params.clusters; ++k)
				action.cluster(mesh, rng);
		}
		if (i < 0)
			continue;

		// basic observables
		res.actionHistory.push_back(action.action(mesh));
		res.magHistory.push_back(action.magnetization(mesh));

		// measure 2pt correlator
		// corr.compute();
		// xt::view(res.c2pt, i) = corr.fullCorr();

		// write config to file
		if (params.filename != "")
		{
			std::vector<hsize_t> shape;
			for (int d : params.geom)
				shape.push_back(d);
			shape.push_back(Action::rep);

			std::string name = fmt::format("/configs/{}", i + 1);
			file.createData(name, shape, params.hdf5Type)
			    .write(mesh.rawConfig());
		}
	}
	pb.finish();

	/** analyze measurements */
	res.reject = (double)action.nReject / (action.nAccept + action.nReject);

	if (params.filename != "")
	{
		file.createData("action_history", {res.actionHistory.size()})
		    .write(res.actionHistory);
		file.createData("mag_history", {res.magHistory.size()})
		    .write(res.magHistory);
		// file.createData("c2pt", s).write(res.c2pt);
	}

	return res;
}

// template scalar_chain_result_t
// runChain<phi4_action>(const scalar_chain_param_t<phi4_action> &);
// template scalar_chain_result_t
// runChain<sigma_action>(const scalar_chain_param_t<sigma_action> &);
template ScalarChainResult
runChain<IsingAction>(ScalarChainParams<IsingAction> const &);
