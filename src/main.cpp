#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

#include <fmt/format.h>

#include "mesh/markov.h"
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
	("betaMin", po::value<double>()->default_value(0.0), "inverse temperature")
	("betaMax", po::value<double>()->default_value(1.0), "")
	("beta2", po::value<std::vector<double>>()->multitoken(), "secondary (usually adjoint) coupling")
	("sweeps", po::value<int>()->default_value(50), "number of sweeps per beta value (half is discarded as warmup)")
	("steps", po::value<int>()->default_value(50), "number of steps of beta")
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
	params.count = params.discard = vm["sweeps"].as<int>() / 2;
	params.sweeps = 1;

	auto betaMin = vm["betaMin"].as<double>();
	auto betaMax = vm["betaMax"].as<double>();
	std::vector<double> beta2s;
	if (vm.count("beta2"))
		beta2s = vm["beta2"].as<std::vector<double>>();
	else
		beta2s.push_back(0.0);
	auto nSteps = vm["steps"].as<int>();

	Gnuplot plotAction;
	plotAction.setRangeX(betaMin, betaMax);
	plotAction.setRangeY(0, 1);

	std::vector<std::vector<double>> pBeta;
	std::vector<std::vector<double>> pAction;

	fmt::print("╔═════════════╤═══════════╤══════╤═══════╗\n");
	fmt::print("║  coupling   │   plaq    │  acc │  corr ║\n");
	fmt::print("╟─────────────┼───────────┼──────┼───────╢\n");

	for (double beta2 : beta2s)
	{
		pBeta.emplace_back();
		pAction.emplace_back();

		for (int i = 0; i < nSteps; ++i)
		{
			auto beta = betaMin + i * (betaMax - betaMin) / (nSteps - 1);

			params.beta = beta;
			params.beta2 = beta2;
			auto res = runChain(params);

			fmt::print("║ {:5.3f} {:5.3f} │ {:9.7f} │ {:4.2f} │ {:5.2f} ║\n",
			           beta, beta2, res.action, 0.0 / 0.0, res.corrTime);

			// plot measurements
			pBeta.back().push_back(beta);
			pAction.back().push_back(res.action);

			plotAction.clear();
			for (size_t i = 0; i < pBeta.size(); ++i)
				plotAction.plotData(
				    pBeta[i], pAction[i],
				    fmt::format("<action> (b2 = {})", beta2s[i]));
		}
	}

	fmt::print("╚═════════════╧═══════════╧══════╧═══════╝\n");
}
