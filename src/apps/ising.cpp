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
#include <omp.h>
#include <random>
#include <vector>

enum class Algo
{
	heat_bath,
	exact_heat_bath,
	swendsen_wang,
	exact_swendsen_wang
};
static const std::map<std::string, Algo> str2algo{
    {"heat-bath", Algo::heat_bath},
    {"exact-heat-bath", Algo::exact_heat_bath},
    {"swendsen-wang", Algo::swendsen_wang},
    {"exact-swendsen-wang", Algo::exact_swendsen_wang}};

int main(int argc, char **argv)
{
	IsingParams params;
	params.geom = {32, 32};
	params.count = 1000;
	params.beta = 0.44068679350977147; // beta_crit for 2D
	bool do_plot = false;
	params.overwrite_existing = false;
	Algo algorithm = Algo::exact_swendsen_wang;
	int num_threads = 0;

	CLI::App app{"Simulate 1D-4D Ising model with a variety of algorithm."};

	// physics options
	app.add_option("--geom", params.geom, "geometry of the lattice");
	app.add_option("--beta", params.beta, "inverse temperature");

	// markov options
	app.add_option("--count", params.count, "number of configs to generate");
	app.add_option("--discard", params.discard,
	               "number of configs to discard for thermalization (only for "
	               "non-exact algos)");
	app.add_option(
	    "--spacing", params.spacing,
	    "number of updates between configs (only for non-exact algos)");
	app.add_option(
	    "--seed", params.seed,
	    "seed for random number generator (default = empty = random)");
	app.add_option("--algorithm", algorithm, "simulation algorithm")
	    ->transform(CLI::CheckedTransformer(str2algo, CLI::ignore_case));
	app.add_option(
	    "--threads,-j", num_threads,
	    "number of worker threads to spawn "
	    "(not implemented for all algorithms). If not given, use OpenMP "
	    "defaults, which can controlled by setting OMP_NUM_THREADS.");

	// output options
	app.add_flag("--plot", do_plot, "plot result using Gnuplot");
	app.add_option("--filename", params.filename,
	               "hdf5 output with one dataset per config. If given a path "
	               "ending with '/', a filename is chosen automatically.");
	app.add_flag("--force", params.overwrite_existing,
	             "allow overwriting existing datafile");

	CLI11_PARSE(app, argc, argv);

	// no seed given -> get a random one
	if (params.seed.empty())
		params.seed = fmt::format("{}", std::random_device()());

	// if no thread count is given, stick to OpenMP's default
	if (num_threads)
		omp_set_num_threads(num_threads);

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
	switch (algorithm)
	{
	case Algo::heat_bath:
		res = run_heat_bath(params);
		break;
	case Algo::exact_heat_bath:
		res = run_exact_heat_bath(params);
		break;
	case Algo::swendsen_wang:
		res = run_swendsen_wang(params);
		break;
	case Algo::exact_swendsen_wang:
		res = run_exact_swendsen_wang(params);
		break;
	default:
		assert(false);
	}

	// double tau = util::correlationTime(res.magnetizationHistory);

	if (do_plot)
	{
		util::Gnuplot().plotData(res.magnetizationHistory, "M");
		// util::Gnuplot().plotData(res.susceptibilityHistory, "chi");
	}

	// analyze

	// fmt::print("beta = {}, <mag> = {}, <|mag|> = {}, corr = {}\n", beta,
	// mag, magAbs, tau);

	// known exact formula for 2D infinite-volume
	// 2D infinite volume exact result (for ordered phase):
	// <|mag|> = pow(1 - pow(sinh(2 * beta), -4), 0.125)
}
