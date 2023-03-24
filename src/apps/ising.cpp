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
	IsingParams params;
	params.geom = {32, 32};
	params.count = 1000;
	params.beta = 0.44068679350977147; // beta_crit for 2D
	bool do_plot = false;
	params.overwrite_existing = false;
	std::string algorithm = "swendsen-wang";

	CLI::App app{"Simulate 1D-4D Ising model "
	             "with a combination of heatbath and cluster updates."};

	// physics options
	app.add_option("--geom", params.geom, "geometry of the lattice");
	app.add_option("--beta", params.beta, "inverse temperature");

	// markov options
	app.add_option("--count", params.count, "number of configs to generate");
	app.add_option("--discard", params.discard,
	               "number of configs to discard for thermalization");
	app.add_option("--spacing", params.spacing,
	               "number of updates between configs");
	app.add_option(
	    "--seed", params.seed,
	    "seed for random number generator (default = empty = random)");
	app.add_option("--algorithm", algorithm,
	               "simulation algorithm (SwendsenWang=default, HeatBath)");

	// output options
	app.add_flag("--plot", do_plot, "plot result");
	app.add_option("--filename", params.filename,
	               "hdf5 output (one dataset per config)");
	app.add_flag("--force", params.overwrite_existing,
	             "overwrite existing data file");

	CLI11_PARSE(app, argc, argv);

	// no seed given -> get a random one
	if (params.seed.empty())
		params.seed = fmt::format("{}", std::random_device()());

	// filename ends with "/" -> automatic filename
	if (params.filename != "" && params.filename.back() == '/')
	{

		std::string name;
		if (params.beta >= 1)
		{
			name = fmt::format("L{}_b{:.3f}", params.geom[0], params.beta);
			std::replace(name.begin(), name.end(), '.', 'p');
		}
		else
			name = fmt::format("L{}_bp{:03}", params.geom[0],
			                   int(std::round(1000 * params.beta)));
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
	fmt::print("starting run with L={}, beta={:.4f}\n", params.geom[0],
	           params.beta);
	if (params.filename != "")
		fmt::print("writing to {}\n", params.filename);

	IsingResults res;
	if (algorithm == "heat-bath")
		res = run_heat_bath(params);
	else if (algorithm == "exact-heat-bath")
		res = run_exact_heat_bath(params);
	else if (algorithm == "swendsen-wang")
		res = run_swendsen_wang(params);
	else if (algorithm == "exact-swendsen-wang")
		res = run_exact_swendsen_wang(params);
	else
	{
		fmt::print("ERROR: unknown algorithm '{}'\n", algorithm);
		return -1;
	}

	// double tau = util::correlationTime(res.magnetizationHistory);

	if (do_plot)
	{
		util::Gnuplot().plotData(res.magnetizationHistory, "M");
		// util::Gnuplot().plotData(res.susceptibilityHistory, "chi");
	}

	// analyze

	// fmt::print("beta = {}, <mag> = {}, <|mag|> = {}, corr = {}\n", beta, mag,
	// magAbs, tau);

	// known exact formula for 2D infinite-volume
	// 2D infinite volume exact result (for ordered phase):
	// <|mag|> = pow(1 - pow(sinh(2 * beta), -4), 0.125)
}
