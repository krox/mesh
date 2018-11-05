#include <cassert>
#include <cmath>
#include <experimental/filesystem>
#include <iostream>
#include <random>
#include <vector>

#include <fmt/format.h>

#include "xtensor/xstrides.hpp"

#include "xtensor/xadapt.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xfunction.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

#include "mesh/topology.h"
#include "scalar/scalar.h"
#include "scalar/sigma.h"
#include "util/fft.h"
#include "util/gnuplot.h"
#include "util/random.h"

#include "boost/program_options.hpp"
namespace po = boost::program_options;

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
}

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
	// clang-format off
	po::options_description desc("Allowed options");
	desc.add_options()
	("help", "this help message")
	("geom", po::value<std::vector<int>>()->multitoken(), "lattice size")
	("beta", po::value<std::vector<double>>()->multitoken(), "inverse coupling ( = 1/T = 1/g_0^2 )")
	("mu", po::value<std::vector<double>>()->multitoken(), "chemical potential")
	("count", po::value<int>()->default_value(1000), "number of gauge-configs to generate")
	("discard", po::value<int>()->default_value(100), "number of gauge-configs to discard (thermalization)")
	("sweeps", po::value<int>()->default_value(10), "number of sweeps between configs")
	("seed", po::value<uint64_t>()->default_value(std::random_device()()), "seed for random number generator")
	("filename", po::value<std::string>()->default_value(""), "hdf5 output")
	("skip-config", "do not include actual configs in output")
	("plot", "show plot of generated ensemble(s)")
	;
	// clang-format on

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help"))
	{
		std::cout << desc << "\n";
		return 1;
	}

	scalar_chain_param_t<sigma_action> param;
	param.geom = vm["geom"].as<std::vector<int>>();
	param.count = vm["count"].as<int>();
	param.discard = vm["discard"].as<int>();
	param.sweeps = vm["sweeps"].as<int>();
	param.seed = vm["seed"].as<uint64_t>();
	param.skipConfig = vm.count("skip-config");
	std::string filename = vm["filename"].as<std::string>();
	bool doPlot = vm.count("plot");

	auto betas = vm["beta"].as<std::vector<double>>();
	auto mus = vm["mu"].as<std::vector<double>>();
	double dim = param.geom.size();

	std::vector<double> plotBeta, plotMu, plotMass, plotAction;

	auto plot = Gnuplot();
	auto plot2 = Gnuplot();

	for (double beta : betas)
	{
		for (double mu : mus)
		{
			// parameters for this run
			param.param.beta = beta;
			param.param.mu = mu;

			if (!filename.empty())
			{
				// automatic filename if only path was given
				if (filename.size() < 3 ||
				    0 != filename.compare(filename.size() - 3, 3, ".h5"))
					param.filename =
					    fmt::format("{}/{}", filename,
					                autoFilename(param.geom, param.param.beta,
					                             param.param.mu));
			}

			// run the chain
			auto res = runChain(param);

			// analyze results
			double mass = analyzeMass(res.c2pt, doPlot && betas.size() == 1 &&
			                                        mus.size() == 1);
			double action = xt::mean(res.actionHistory)();
			double signReal = xt::mean(xt::cos(res.phaseAngle))();
			double signImag = xt::mean(xt::sin(res.phaseAngle))();
			fmt::print("beta = {}, mu = {}, mass = {}, <sign> = {} + {}i, mL = "
			           "{}, reject = {}\n",
			           beta, mu, mass, signReal, signImag, mass * param.geom[0],
			           res.reject);

			plotBeta.push_back(beta);
			plotMu.push_back(mu);
			plotMass.push_back(mass);
			plotAction.push_back(action);

			if (doPlot && plotMass.size() >= 2)
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
			}

			if (doPlot && plotAction.size() >= 2)
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
