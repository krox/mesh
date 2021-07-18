#include "scalar/ising.h"
#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "mesh/topology.h"
#include "util/gnuplot.h"
#include "util/random.h"
#include "util/sampler.h"
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char **argv)
{
	ScalarChainParams<IsingAction> params;
	params.geom = {32, 32};
	params.count = 1000;
	params.discard = 100;
	params.sweeps = 2;
	params.clusters = 1;
	params.seed = (uint64_t)-1;
	params.actionParams.beta = 0.44068679350977147; // beta_crit for 2D
	bool do_plot = false;
	params.overwrite_existing = false;

	CLI::App app{"Simulate 1D-4D Ising model "
	             "with a combination of heatbath and cluster updates."};

	// physics options
	app.add_option("--geom", params.geom, "geometry of the lattice");
	app.add_option("--beta", params.actionParams.beta, "inverse temperature");

	// markov options
	app.add_option("--count", params.count, "number of configs to generate");
	app.add_option("--discard", params.discard,
	               "number of configs to discard for thermalization");
	app.add_option("--sweeps", params.sweeps,
	               "number of heatbath sweeps between configs");
	app.add_option("--clusters", params.sweeps,
	               "number of cluster flips per heatbath sweeps");
	app.add_option("--seed", params.seed,
	               "seed for random number generator (default = random)");

	// output options
	app.add_flag("--plot", do_plot, "plot result");
	app.add_option("--filename", params.filename,
	               "hdf5 output (one dataset per config)");
	app.add_flag("--force", params.overwrite_existing,
	             "overwrite existing data file");

	CLI11_PARSE(app, argc, argv);

	// no seed given -> get a random one
	if (params.seed == (uint64_t)-1)
		params.seed = std::random_device()();

	// filename ends with "/" -> automatic filename
	if (params.filename != "" && params.filename.back() == '/')
	{
		std::string name = fmt::format("L{}_b{:.4f}", params.geom[0],
		                               params.actionParams.beta);
		std::replace(name.begin(), name.end(), '.', 'p');
		params.filename = fmt::format("{}{}.h5", params.filename, name);
	}

	// file aleady exists -> abort
	if (params.filename != "" && !params.overwrite_existing &&
	    std::filesystem::exists(params.filename))
	{
		fmt::print("{} already exists. aborting.\n", params.filename);
		return 0;
	}

	// run a chain
	auto res = runChain(params);

	double tau = util::correlationTime(res.magHistory);

	if (do_plot)
		util::Gnuplot().plotData(res.magHistory);

	// analyze

	// fmt::print("beta = {}, <mag> = {}, <|mag|> = {}, corr = {}\n", beta, mag,
	// magAbs, tau);

	// known exact formula for 2D infinite-volume
	// 2D infinite volume exact result (for ordered phase):
	// <|mag|> = pow(1 - pow(sinh(2 * beta), -4), 0.125)
}
