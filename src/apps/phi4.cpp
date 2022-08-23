#include "scalar/phi4.h"
#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "mesh/topology.h"
#include "scalar/scalar.h"
#include "util/gnuplot.h"
#include "util/random.h"
#include <cassert>
#include <cmath>
#include <experimental/filesystem>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char **argv)
{

	scalar_chain_param_t<phi4_action> param;
	param.geom = {128};
	param.count = 1000;
	param.discard = 100;
	param.sweeps = 10;
	param.clusters = 0;
	param.seed = (uint64_t)-1;
	std::vector<double> masses = {};
	double mass_min = 0.0;
	double mass_max = 3.0;
	int mass_count = 20;
	bool do_plot = false;

	CLI::App app{"Simulate scalar phi^4 model."};
	// physics options
	app.add_option("--geom", param.geom, "geometry of the lattice");
	app.add_option("--beta", masses, "coupling constant");

	app.add_option("--mass-min", mass_min, "coupling constant");
	app.add_option("--mass-max", mass_max, "coupling constant");
	app.add_option("--mass-count", mass_count, "coupling constant");

	// markov options
	app.add_option("--count", param.count, "number of configs to generate");
	app.add_option("--discard", param.discard,
	               "number of configs to discard for thermalization");
	app.add_option("--sweeps", param.sweeps,
	               "number of heatbath sweeps between configs");
	app.add_option("--seed", param.seed,
	               "seed for random number generator (default = random)");

	// output options
	app.add_flag("--plot", do_plot, "plot result");
	app.add_flag("--filename", param.filename,
	             "hdf5 output (one dataset per config)");

	CLI11_PARSE(app, argc, argv);

	// no explicit (list of) beta values -> make a sweep
	if (masses.empty())
		for (int i = 0; i < mass_count; ++i)
			masses.push_back(mass_min + 1.0 * i / (mass_count - 1) *
			                                (mass_max - mass_min));

	// no seed given -> get a random one
	if (param.seed == (uint64_t)-1)
		param.seed = std::random_device()();

	if (param.filename != "" && masses.size() != 1)
	{
		fmt::print("ERROR: HDF5 output only possible for single beta value\n");
		return -1;
	}

	for (double mass : masses)
	{
		// run a chain
		param.param.mass = mass;
		auto res = runChain(param);

		// analyze results
		// analyzeMass(res.c2pt, true);
	}
}
