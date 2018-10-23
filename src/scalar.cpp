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
		xt::xarray<double> mean = xt::mean(res.c2pt, {0});
		double c2pt0 = mean[0];

		// plot results
		Gnuplot().setLogScaleY().setRangeX(0, 30).plotData(mean).plotFunction(
		    [&](double x) { return c2pt0 * exp(-x * mass); }, 0, 30,
		    "free  theory");
	}
}
