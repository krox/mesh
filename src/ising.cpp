#include "scalar/ising.h"
#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "mesh/topology.h"
#include "util/gnuplot.h"
#include "util/random.h"
#include "util/sampler.h"
#include <cassert>
#include <cmath>
#include <experimental/filesystem>
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
	std::vector<double> betas = {};
	double beta_min = 0.0;
	double beta_max = 1.0;
	int beta_count = 20;
	bool do_plot = false;

	CLI::App app{"Simulate 2D Ising model with a combination of heatbath and "
	             "cluster updates."};
	// physics options
	app.add_option("--geom", params.geom, "geometry of the lattice");
	app.add_option("--beta", betas, "coupling constant");
	app.add_option("--beta-min", beta_min, "coupling constant");
	app.add_option("--beta-max", beta_max, "coupling constant");
	app.add_option("--beta-count", beta_count, "coupling constant");

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
	app.add_flag("--filename", params.filename,
	             "hdf5 output (one dataset per config)");

	CLI11_PARSE(app, argc, argv);

	// no explicit (list of) beta values -> make a sweep
	if (betas.empty())
		for (int i = 0; i < beta_count; ++i)
			betas.push_back(beta_min +
			                1.0 * i / (beta_count - 1) * (beta_max - beta_min));

	// no seed given -> get a random one
	if (params.seed == (uint64_t)-1)
		params.seed = std::random_device()();

	if (params.filename != "" && betas.size() != 1)
	{
		fmt::print("ERROR: HDF5 output only possible for single beta value\n");
		return -1;
	}

	std::vector<double> plotBeta, plotMag, plotMagAbs;

	for (double beta : betas)
	{
		// run a chain
		params.actionParams.beta = beta;
		auto res = runChain(params);

		if (betas.size() == 1 && do_plot)
			util::Gnuplot().plotData(res.magHistory);

		// analyze
		double mag = util::mean(res.magHistory);
		double magAbs = util::mean_abs(res.magHistory);
		double tau = util::correlationTime(res.magHistory);
		fmt::print("beta = {}, <mag> = {}, <|mag|> = {}, corr = {}\n", beta,
		           mag, magAbs, tau);
		plotBeta.push_back(beta);
		plotMag.push_back(mag);
		plotMagAbs.push_back(magAbs);

		params.seed += 1; // change seed for next beta
	}

	if (betas.size() >= 2 && do_plot)
	{
		auto plot = util::Gnuplot();
		plot.plotData(plotBeta, plotMag, "<mag>");
		plot.plotData(plotBeta, plotMagAbs, "<|mag|>");

		// known exact formula for 2D infinite-volume
		if (params.geom.size() == 2)
		{
			plot.plotFunction(
			    [](double beta) {
				    return pow(1 - pow(sinh(2 * beta), -4), 0.125);
			    },
			    0.44068679350977147, betas.back());
			plot.plotFunction([](double) { return 0.0; }, betas.front(),
			                  0.44068679350977147);
		}
	}
}
