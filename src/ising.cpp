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
#include "scalar/ising.h"
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
	("beta", po::value<std::vector<double>>()->multitoken(), "interaction")
	("betaMin", po::value<double>()->default_value(0.0), "interaction")
	("betaMax", po::value<double>()->default_value(1.0), "interaction")
	("betaCount", po::value<int>()->default_value(20), "interaction")
	("count", po::value<int>()->default_value(1000), "number of gauge-configs to generate")
	("discard", po::value<int>()->default_value(100), "number of gauge-configs to discard (thermalization)")
	("sweeps", po::value<int>()->default_value(5), "number of sweeps between configs")
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

	scalar_chain_param_t<ising_action> param;
	param.geom = vm["geom"].as<std::vector<int>>();
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

	std::vector<double> plotBeta, plotMag;

	for (double beta : betas)
	{
		// run a chain
		param.param.beta = beta;
		auto res = runChain(param);

		if (betas.size() == 1)
		{
			Gnuplot().plotData(res.magHistory);
		}

		// analyze
		double mag = xt::mean(xt::abs(res.magHistory))();
		fmt::print("beta = {}, |mag| = {}\n", beta, mag);
		plotBeta.push_back(beta);
		plotMag.push_back(mag);

		param.seed += 1; // change seed for next beta
	}

	if (betas.size() >= 2)
	{
		Gnuplot()
		    .plotData(plotBeta, plotMag)
		    .plotFunction(
		        [](double beta) {
			        return pow(1 - pow(sinh(2 * beta), -4), 0.125);
		        },
		        0.44068679350977147, betas.back());
	}
}
