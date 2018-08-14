#include <cassert>
#include <cmath>
#include <experimental/filesystem>
#include <iostream>
#include <random>

#include <fmt/format.h>

#include "mesh/topology.h"
#include "scalar/scalar.h"
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
	("kappa", po::value<double>()->default_value(0.0), "hopping parameter")
	("lambda", po::value<double>()->default_value(0.0), "quartic coupling")
	("count", po::value<int>()->default_value(1000), "number of gauge-configs to generate")
	("discard", po::value<int>()->default_value(100), "number of gauge-configs to discard (thermalization)")
	("sweeps", po::value<int>()->default_value(10), "number of sweeps between configs")
	("or", po::value<int>()->default_value(0), "number of OR steps per heat bath")
	("seed", po::value<uint64_t>()->default_value(std::random_device()()), "seed for random number generator (same for all betas)")
	("file", po::value<std::string>()->default_value(""), "file output (HDF5 format)")
	("plot", "show plot of generated ensemble")
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

	scalar::ChainParams params;
	params.geom = vm["geom"].as<std::vector<int>>();
	params.count = vm["count"].as<int>();
	params.discard = vm["discard"].as<int>();
	params.sweeps = vm["sweeps"].as<int>();
	params.overrelax = vm["or"].as<int>();
	params.seed = vm["seed"].as<uint64_t>();
	params.kappa = vm["kappa"].as<double>();
	params.lambda = vm["lambda"].as<double>();
	params.filename = vm["file"].as<std::string>();
	bool doPlot = vm.count("plot");

	if (!params.filename.empty())
	{
		// automatic filename if only path was given
		if (params.filename.size() < 3 ||
		    0 != params.filename.compare(params.filename.size() - 3, 3, ".h5"))
			params.filename += '/' + params.autoFilename();

		// skip if file already exists
		if (std::experimental::filesystem::exists(params.filename))
		{
			fmt::print("chain '{}' already exists\n", params.filename);
			return 0;
		}
	}

	fmt::print("starting chain '{}'\n", params.filename);
	auto res = scalar::runChain(params);

	fmt::print("kappa = {}\n", params.kappa);
	fmt::print("lambda = {}\n", params.lambda);
	fmt::print("geom =");
	for (auto g : params.geom)
		fmt::print(" {}", g);
	fmt::print("\n");
	fmt::print("action = {} (theory = {})\n", res.action,
	           0.5 / (1 - params.kappa * params.kappa));
	fmt::print("corr-time = {}\n", res.corrTime);

	if (doPlot)
	{
		Gnuplot()
		    .setRangeX(0, res.actionHistory.size())
		    .plotData(
		        res.actionHistory,
		        fmt::format("<x^2> (k={}, l={})", params.kappa, params.lambda));

		auto tau = res.corrTime * params.sweeps;
		int T = std::min((int)(tau * 5), (int)res.actionHistory.size() / 10);
		Gnuplot()
		    .plotData(autocorrelation(res.actionHistory, T))
		    .plotFunction([=](double x) { return exp(-x / tau); }, 0, T,
		                  fmt::format("corr-time*{} = {}", params.sweeps, tau));
	}
}
