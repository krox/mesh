#include "scalar/sigma.h"
#include "CLI/CLI.hpp"
#include "mesh/topology.h"
#include "scalar/scalar.h"
#include "util/gnuplot.h"
#include "util/random.h"
#include <cassert>
#include <cmath>
#include <experimental/filesystem>
#include <fmt/format.h>
#include <iostream>
#include <random>
#include <vector>

/*
xt::xarray<double> getMom0(xt::xarray<double> c2pt)
{
    while (c2pt.dimension() > 2)
        c2pt = xt::sum(c2pt, {1}) / c2pt.shape()[1];
    return c2pt;
}

double analyzeMass(const xt::xarray<double> &c2pt_full, bool plot)
{
    // spatial momentum zero
    xt::xarray<double> c2pt = getMom0(c2pt_full);
    xt::xarray<double> mean = xt::mean(c2pt, {0});
    xt::xarray<double> err =
        xt::sqrt(xt::mean((c2pt - mean) * (c2pt - mean), {0})) /
        sqrt(c2pt.shape()[0]);

    // exponential fit using log-trick
    std::vector<double> xs, ys, yse;
    for (size_t t = 0; t <= mean.shape()[0] / 4; ++t)
    {
        if (mean(t) < 2 * err(t))
            break;
        xs.push_back(t);
        ys.push_back(log(mean(t)));
        yse.push_back(err(t) / mean(t));
    }
    auto fit = LinearFit(xs, ys, yse);
    double a0 = exp(fit.a);
    double mass = -fit.b;

    // plot 2pt function
    if (plot)
    {
        Gnuplot()
            .setLogScaleY()
            .setRangeX(0, (int)xs.size() + 2)
            .plotErrorbar(mean, err)
            .plotFunction([&](double x) { return a0 * exp(-x * mass); }, 0,
                          (int)xs.size(), fmt::format("mass = {}", mass));
    }
    return mass;
}*/

std::string autoFilename(const std::vector<int> geom, double beta, double mu)
{
	if (mu == 0)
		return fmt::format("O3.{}x{}.b{}.h5", geom[0], geom[1],
		                   (int)(beta * 1000));
	else
		return fmt::format("O3.{}x{}.b{}.m{}.h5", geom[0], geom[1],
		                   (int)(beta * 1000), (int)(mu * 1000));
}

int main(int argc, char **argv)
{
	scalar_chain_param_t<sigma_action> param;
	param.geom = {32, 32};
	param.count = 1000;
	param.discard = 100;
	param.sweeps = 2;
	param.clusters = 0;
	param.seed = (uint64_t)-1;
	std::vector<double> betas = {};
	std::vector<double> mus = {0.0};
	double beta_min = 0.0;
	double beta_max = 3.0;
	int beta_count = 20;
	bool do_plot = false;

	CLI::App app{"Simulate the O(3) non-linear sigma model."};
	// physics options
	app.add_option("--geom", param.geom, "geometry of the lattice");
	app.add_option("--beta", betas, "coupling constant");

	app.add_option("--beta-min", beta_min, "coupling constant");
	app.add_option("--beta-max", beta_max, "coupling constant");
	app.add_option("--beta-count", beta_count, "coupling constant");

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
	if (betas.empty())
		for (int i = 0; i < beta_count; ++i)
			betas.push_back(beta_min +
			                1.0 * i / (beta_count - 1) * (beta_max - beta_min));

	// no seed given -> get a random one
	if (param.seed == (uint64_t)-1)
		param.seed = std::random_device()();

	if (param.filename != "" && betas.size() != 1)
	{
		fmt::print("ERROR: HDF5 output only possible for single beta value\n");
		return -1;
	}

	double dim = param.geom.size();

	std::vector<double> plotBeta, plotMu, plotAction;

	auto plot = util::Gnuplot();
	auto plot2 = util::Gnuplot();

	for (double beta : betas)
	{
		for (double mu : mus)
		{
			// parameters for this run
			param.param.beta = beta;
			param.param.mu = mu;

			// run the chain
			auto res = runChain(param);

			// analyze results
			/*double mass = analyzeMass(res.c2pt, do_plot && betas.size() == 1
			   && mus.size() == 1);*/
			double action = util::mean(res.actionHistory);
			fmt::print("beta = {}, mu = {}, action = {}, reject = {}\n", beta,
			           mu, action, res.reject);

			plotBeta.push_back(beta);
			plotMu.push_back(mu);
			// plotMass.push_back(mass);
			plotAction.push_back(action);

			/*if (do_plot && plotMass.size() >= 2)
			{
			    plot.clear();
			    plot.plotData(plotBeta, plotMass);

			    if (dim == 1)
			    {
			        plot.plotFunction(
			            [&](double b) {
			                return -log(b / 3.0 - b * b * b / 45.0);
			            },
			            betas.front(), 2.5, "strong coupling");
			        plot.plotFunction(
			            [&](double b) { return -log(1.0 - 1.0 / b); }, 1.5,
			            betas.back(), "weak coupling");
			    }
			}*/

			if (do_plot && plotAction.size() >= 2)
			{
				plot2.clear();
				plot2.plotData(plotBeta, plotAction, "action density");

				if (dim == 2)
				{
					plot2.plotFunction(
					    [&](double b) {
						    double y = cosh(b) / sinh(b) - 1 / b;
						    return 4 - 4 * y - 8 * y * y * y -
						           48 / 5.0 * y * y * y * y * y;
					    },
					    betas.front(), 1.3, "strong coupling");
					plot2.plotFunction(
					    [&](double b) {
						    return 2 / b + 0.25 / b / b + 0.156 / b / b / b;
					    },
					    1.2, betas.back(), "weak coupling");
				}
			}
		}
	}
}
