#include <cassert>
#include <cmath>
#include <experimental/filesystem>
#include <iostream>
#include <random>
#include <vector>

#include <fmt/format.h>

#include "xtensor/xstrides.hpp"

#include "xtensor/xmath.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

#include "mesh/topology.h"
#include "scalar/phi4.h"
#include "scalar/scalar.h"
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

int main(int argc, char **argv)
{
	// clang-format off
	po::options_description desc("Allowed options");
	desc.add_options()
	("help", "this help message")
	("geom", po::value<std::vector<int>>()->multitoken(), "lattice size")
	("mass", po::value<std::vector<double>>()->multitoken(), "bare lattice mass")
	("count", po::value<int>()->default_value(1000), "number of gauge-configs to generate")
	("discard", po::value<int>()->default_value(100), "number of gauge-configs to discard (thermalization)")
	("sweeps", po::value<int>()->default_value(10), "number of sweeps between configs")
	("seed", po::value<uint64_t>()->default_value(std::random_device()()), "seed for random number generator")
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

	scalar_chain_param_t<phi4_action> param;
	param.geom = vm["geom"].as<std::vector<int>>();
	param.count = vm["count"].as<int>();
	param.discard = vm["discard"].as<int>();
	param.sweeps = vm["sweeps"].as<int>();
	param.seed = vm["seed"].as<uint64_t>();

	auto masses = vm["mass"].as<std::vector<double>>();

	for (double mass : masses)
	{
		// run a chain
		param.param.mass = mass;
		auto res = runChain(param);

		// analyze results
		analyzeMass(res.c2pt, true);
	}
}
