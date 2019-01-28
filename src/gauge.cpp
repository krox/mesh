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

#include "gauge/gauge.h"
#include "gauge/markov.h"
#include "gauge/wilson.h"
#include "groups/su2.h"
#include "groups/u1.h"
#include "groups/z2.h"
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
	("group,g", po::value<std::string>()->default_value("su2"), "gauge group")
	("beta", po::value<std::vector<double>>()->multitoken(), "interaction")
	("betaMin", po::value<double>()->default_value(0.0), "interaction")
	("betaMax", po::value<double>()->default_value(3.0), "interaction")
	("betaCount", po::value<int>()->default_value(31), "interaction")
	("count", po::value<int>()->default_value(100), "number of gauge-configs to generate")
	("discard", po::value<int>()->default_value(100), "number of gauge-configs to discard (thermalization)")
	("sweeps", po::value<int>()->default_value(1), "number of heatbath sweeps between configs")
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

	GaugeChainParams param;
	param.geom = vm["geom"].as<std::vector<int>>();
	param.group = vm["group"].as<std::string>();
	param.count = vm["count"].as<int>();
	param.discard = vm["discard"].as<int>();
	param.sweeps = vm["sweeps"].as<int>();
	param.seed = vm["seed"].as<uint64_t>();

	std::vector<double> betas;
	if (vm.count("beta"))
		betas = vm["beta"].as<std::vector<double>>();
	else
	{
		double betaMin = vm["betaMin"].as<double>();
		double betaMax = vm["betaMax"].as<double>();
		int n = vm["betaCount"].as<int>();

		for (int i = 0; i < n; ++i)
			betas.push_back(betaMin + 1.0 * i / (n - 1) * (betaMax - betaMin));
	}

	std::vector<double> plotBeta, plotPlaq, plotCorr;

	for (double beta : betas)
	{
		// run a chain
		WilsonActionParams actionParams;
		actionParams.beta = beta;
		auto res = runChain(param, actionParams);

		if (betas.size() == 1)
		{
			Gnuplot().plotData(res.plaqHistory, "<plaq>");
			Gnuplot().plotData(res.topHistory, "<Q_{top}>");
		}

		// analyze
		double plaq = xt::mean(res.plaqHistory)();
		double tau = correlationTime(res.plaqHistory);
		fmt::print("beta = {}, plaq = {}, corr = {}\n", beta, plaq, tau);
		plotBeta.push_back(beta);
		plotPlaq.push_back(plaq);
		plotCorr.push_back(tau);

		param.seed += 1; // change seed for next beta
	}

	if (betas.size() >= 2)
	{
		Gnuplot().setRangeY(0, 1).plotData(plotBeta, plotPlaq, "<plaq>");
		Gnuplot().plotData(plotBeta, plotCorr);
	}
}
