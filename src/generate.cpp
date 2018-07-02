#include <cassert>
#include <cmath>
#include <experimental/filesystem>
#include <iostream>
#include <random>

#include <fmt/format.h>

#include "mesh/gauge_action.h"
#include "mesh/gauge_fixing.h"
#include "mesh/markov.h"
#include "mesh/mesh.h"
#include "mesh/su2.h"
#include "mesh/u1.h"
#include "mesh/z2.h"
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
	("group,g", po::value<std::string>()->default_value("u1"), "gauge group")
	("geom", po::value<std::vector<int>>()->multitoken(), "lattice size")
	("beta", po::value<double>()->default_value(1.0), "inverse coupling (linear term)")
	("beta2", po::value<double>()->default_value(0.0), "inverse coupling (quadratric term)")
	("count", po::value<int>()->default_value(100), "number of gauge-configs to generate")
	("discard", po::value<int>()->default_value(0), "number of gauge-configs to discard (thermalization)")
	("sweeps", po::value<int>()->default_value(20), "number of sweeps between configs")
	("or", po::value<int>()->default_value(3), "number of OR steps per heat bath")
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

	ChainParams params;
	params.group = vm["group"].as<std::string>();
	params.geom = vm["geom"].as<std::vector<int>>();
	params.count = vm["count"].as<int>();
	params.discard = vm["discard"].as<int>();
	params.sweeps = vm["sweeps"].as<int>();
	params.overrelax = vm["or"].as<int>();
	params.seed = vm["seed"].as<uint64_t>();
	params.beta = vm["beta"].as<double>();
	params.beta2 = vm["beta2"].as<double>();
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
	auto res = runChain(params);

	fmt::print("group = {}\n", params.group);
	fmt::print("beta = {}\n", params.beta);
	fmt::print("beta2 = {}\n", params.beta2);
	fmt::print("geom =");
	for (auto g : params.geom)
		fmt::print(" {}", g);
	fmt::print("\n");
	fmt::print("action = {}\n", res.action);
	fmt::print("corr-time = {}\n", res.corrTime);

	if (doPlot)
	{
		Gnuplot()
		    .setRangeX(0, res.actionHistory.size())
		    .plotData(res.actionHistory,
		              fmt::format("<action> (b={}, b2={})", params.beta,
		                          params.beta2));

		auto tau = res.corrTime * params.sweeps;
		int T = std::min((int)(tau * 5), (int)res.actionHistory.size() / 10);
		Gnuplot()
		    .plotData(autocorrelation(res.actionHistory, T))
		    .plotFunction([=](double x) { return exp(-x / tau); }, 0, T,
		                  fmt::format("corr-time*{} = {}", params.sweeps, tau));
	}
}
