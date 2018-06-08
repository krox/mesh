#include <cassert>
#include <cmath>
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
	("n", po::value<int>()->default_value(8), "lattice size")
	("beta", po::value<std::vector<double>>()->multitoken(), "inverse coupling (linear term)")
	("beta2", po::value<std::vector<double>>()->multitoken(), "inverse coupling (quadratric term)")
	("count", po::value<int>()->default_value(100), "number of gauge-configs to generate")
	("discard", po::value<int>()->default_value(0), "number of gauge-configs to discard (thermalization)")
	("sweeps", po::value<int>()->default_value(20), "number of sweeps between configs")
	("seed", po::value<uint64_t>()->default_value(std::random_device()()), "seed for random number generator (same for all betas)")
	("path", po::value<std::string>()->default_value(""), "path for file output (HDF5 format)")
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
	params.n = vm["n"].as<int>();
	params.count = vm["count"].as<int>();
	params.discard = vm["discard"].as<int>();
	params.sweeps = vm["sweeps"].as<int>();
	params.seed = vm["seed"].as<uint64_t>();

	assert(vm.count("beta"));
	assert(vm.count("beta2"));
	auto betas = vm["beta"].as<std::vector<double>>();
	auto beta2s = vm["beta2"].as<std::vector<double>>();

	bool doPlot = vm.count("plot");
	auto path = vm["path"].as<std::string>();

	fmt::print("╔═════════════╤═══════════╤══════╤═══════╗\n");
	fmt::print("║  coupling   │   plaq    │  acc │  corr ║\n");
	fmt::print("╟─────────────┼───────────┼──────┼───────╢\n");

	for (double beta2 : beta2s)
	{
		for (double beta : betas)
		{
			params.beta = beta;
			params.beta2 = beta2;

			if (path != "")
			{
				if (beta2 == 0)
					params.filename =
					    fmt::format("{}/{}.p{}.b{}.h5", path, params.group,
					                params.n, (int)(beta * 1000));
				else
					params.filename = fmt::format(
					    "{}/{}.p{}.b{}.b{}.h5", path, params.group, params.n,
					    (int)(beta * 1000), (int)(beta2 * 1000));
			}

			auto res = runChain(params);

			fmt::print("║ {:5.3f} {:5.3f} │ {:9.7f} │ {:4.2f} │ {:5.2f} ║\n",
			           beta, beta2, res.action, 0.0 / 0.0, res.corrTime);

			if (doPlot)
			{
				Gnuplot()
				    .setRangeX(params.sweeps * params.discard,
				               res.actionHistory.size())
				    .plotData(
				        res.actionHistory,
				        fmt::format("<action> (b={}, b2={})", beta, beta2));
			}
		}
	}

	fmt::print("╚═════════════╧═══════════╧══════╧═══════╝\n");
}
