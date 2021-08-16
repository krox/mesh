#include "scalar/ising.h"

#include "mesh/topology.h"
#include "util/hdf5.h"
#include "util/progressbar.h"
#include "util/random.h"
#include "util/unionfind.h"
#include <cassert>
#include <fmt/format.h>
#include <random>
#include <vector>

namespace {
double isingAction(Topology const &top, std::vector<int8_t> const &field)
{
	double sum = 0.0;
	for (auto const &link : top.links)
		sum += field[link.from] * field[link.to];
	return sum;
}

double isingMagnetization(std::vector<int8_t> const &field)
{
	double sum = 0.0;
	for (auto s : field)
		sum += s;
	return sum;
}
} // namespace

// TODO: some factoring to remove some code duplication between
//       runHeatBath() and runSwendsenWang()

IsingResults runHeatBath(const IsingParams &params)
{
	// state of the simulation
	auto top = Topology::lattice(params.geom);
	auto field = std::vector<int8_t>(top.nSites());
	util::xoshiro256 rng(params.seed);

	// results
	util::DataFile file;
	IsingResults res;
	// Possible values of the spins are 1,-1. There is no clean way in hdf5
	// to encode that in a single bit, and we dont want to deviate from
	// conventions in the analysis code. Therefore we take 2 bits.
	auto hdf5Type = H5Tcopy(H5T_NATIVE_INT8);
	H5Tset_precision(hdf5Type, 2);

	if (params.filename != "")
	{
		file =
		    util::DataFile::create(params.filename, params.overwrite_existing);

		// physical parameters
		file.setAttribute("beta", params.beta);
		file.setAttribute("geometry", params.geom);

		// simulation parameters
		file.setAttribute("markov_count", params.count);
		file.setAttribute("markov_discard", params.discard);
		file.setAttribute("markov_spacing", params.spacing);

		file.makeGroup("/configs");
	}

	for (auto &x : field)
		x = -1;

	auto pb =
	    util::ProgressBar((params.count + params.discard) * params.spacing);
	for (int iter = -params.discard * params.spacing;
	     iter < params.count * params.spacing; ++iter, ++pb)
	{
		pb.show();

		// one sweep of single-site heat bath
		for (int i = 0; i < top.nSites(); ++i)
		{
			// sum of neighbors
			double rho = 0;
			for (auto const &link : top.graph[i])
				rho += field[link.to];
			rho *= params.beta;

			// probability
			double p = exp(rho) / (exp(rho) + exp(-rho));
			field[i] = std::bernoulli_distribution(p)(rng) ? 1.0 : -1.0;
		}

		// measure observables
		if (iter >= 0)
		{
			res.actionHistory.push_back(params.beta * isingAction(top, field));
			res.magnetizationHistory.push_back(isingMagnetization(field));
		}

		// write config to file
		if (params.filename != "" && iter >= 0 &&
		    (iter + 1) % params.spacing == 0)
		{
			std::vector<hsize_t> shape;
			for (int d : params.geom)
				shape.push_back(d);

			std::string name = fmt::format("/configs/{}", iter + 1);
			file.createData(name, shape, hdf5Type).write(field);
		}
	}
	pb.finish();

	if (params.filename != "")
	{
		file.writeData("action_history", res.actionHistory);
		file.writeData("magnetization_history", res.actionHistory);
	}
	return res;
}

IsingResults runSwendsenWang(const IsingParams &params)
{
	// state of the simulation
	auto top = Topology::lattice(params.geom);
	auto field = std::vector<int8_t>(top.nSites());
	util::xoshiro256 rng(params.seed);

	// stuff needed specifically for Swendsen-Wang
	auto dist = std::bernoulli_distribution(1.0 - exp(-2 * params.beta));
	auto coin = std::bernoulli_distribution(0.5);
	util::UnionFind uf(top.nSites());

	// results
	util::DataFile file;
	IsingResults res;
	// Possible values of the spins are 1,-1. There is no clean way in hdf5
	// to encode that in a single bit, and we dont want to deviate from
	// conventions in the analysis code. Therefore we take 2 bits.
	auto hdf5Type = H5Tcopy(H5T_NATIVE_INT8);
	H5Tset_precision(hdf5Type, 2);

	if (params.filename != "")
	{
		file =
		    util::DataFile::create(params.filename, params.overwrite_existing);

		// physical parameters
		file.setAttribute("beta", params.beta);
		file.setAttribute("geometry", params.geom);

		// simulation parameters
		file.setAttribute("markov_count", params.count);
		file.setAttribute("markov_discard", params.discard);
		file.setAttribute("markov_spacing", params.spacing);

		file.makeGroup("/configs");
	}

	for (auto &x : field)
		x = -1;

	auto pb =
	    util::ProgressBar((params.count + params.discard) * params.spacing);
	for (int iter = -params.discard * params.spacing;
	     iter < params.count * params.spacing; ++iter, ++pb)
	{
		pb.show();

		double susceptibility = 0.0;

		uf.clear();
		for (auto &link : top.links)
			if (field[link.from] == field[link.to] && dist(rng))
				uf.join(link.from, link.to);
		for (int i = 0; i < top.nSites(); ++i)
			if (i == uf.root(i))
			{
				field[i] = coin(rng) ? 1 : -1;

				// do some cluster-based measurements
				auto size = (double)uf.compSize(i);
				susceptibility += size * size;
			}
		for (int i = 0; i < top.nSites(); ++i)
			field[i] = field[uf.root(i)];

		// measure observables
		if (iter >= 0)
		{
			res.actionHistory.push_back(params.beta * isingAction(top, field));
			res.magnetizationHistory.push_back(isingMagnetization(field));
			susceptibility /= top.nSites();
			res.susceptibilityHistory.push_back(susceptibility);
		}

		// write config to file
		if (params.filename != "" && iter >= 0 &&
		    (iter + 1) % params.spacing == 0)
		{
			std::vector<hsize_t> shape;
			for (int d : params.geom)
				shape.push_back(d);

			std::string name = fmt::format("/configs/{}", iter + 1);
			file.createData(name, shape, hdf5Type).write(field);
		}
	}
	pb.finish();

	if (params.filename != "")
	{
		file.writeData("action_history", res.actionHistory);
		file.writeData("magnetization_history", res.actionHistory);
		file.writeData("susceptibility_history", res.susceptibilityHistory);
	}
	return res;
}
