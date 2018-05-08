#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

#include <fmt/format.h>

#include "mesh/gauge_action.h"
#include "mesh/gauge_fixing.h"
#include "mesh/mesh.h"
#include "mesh/su2.h"
#include "mesh/u1.h"
#include "mesh/z2.h"

#include "util/gnuplot.h"
#include "util/random.h"

#include "boost/program_options.hpp"
namespace po = boost::program_options;

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/density.hpp>
#include <boost/accumulators/statistics/stats.hpp>
using namespace boost::accumulators;

int main(int argc, char **argv)
{
	// clang-format off
	po::options_description desc("Allowed options");
	desc.add_options()
	("help", "this help message")
	("n", po::value<int>()->default_value(8), "lattice size")
	("betaMin", po::value<double>()->default_value(0.0), "inverse temperature")
	("betaMax", po::value<double>()->default_value(1.0), "")
	("beta2", po::value<std::vector<double>>()->multitoken(), "secondary (usually adjoint) coupling")
	("sweeps", po::value<int>()->default_value(20), "number of sweeps between measurements")
	("steps", po::value<int>()->default_value(50), "number of steps of beta")
	("gaugefix", po::value<bool>()->default_value(false), "do gauge-fixing and measure average link")
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

	auto n = vm["n"].as<int>();
	auto betaMin = vm["betaMin"].as<double>();
	auto betaMax = vm["betaMax"].as<double>();
	std::vector<double> beta2s;
	if (vm.count("beta2"))
		beta2s = vm["beta2"].as<std::vector<double>>();
	else
		beta2s.push_back(0.0);
	auto nSweeps = vm["sweeps"].as<int>();
	auto nSteps = vm["steps"].as<int>();
	auto doGF = vm["gaugefix"].as<bool>();

	Gnuplot plotAction, plotLink;
	plotAction.setRangeX(betaMin, betaMax);
	plotAction.setRangeY(0, 1);
	plotLink.setRangeX(betaMin, betaMax);
	plotLink.setRangeY(0, 1);

	std::vector<std::vector<double>> pBeta;
	std::vector<std::vector<double>> pAction, pLink;

	xoroshiro128plus rng{std::random_device()()};

	for (double beta2 : beta2s)
	{
		pBeta.emplace_back();
		pAction.emplace_back();
		pLink.emplace_back();

		for (int i = 0; i < nSteps; ++i)
		{
			double beta = betaMin + i * (betaMax - betaMin) / (nSteps - 1);

			auto m = Mesh<U1>(Topology::lattice4D(n));
			auto ga = GaugeAction(m);

			for (int i = 0; i < nSweeps; ++i)
				ga.thermalize(rng, beta, beta2);

			// measure average action
			double loop4 = ga.loop4();

			// measure average link (requires gauge fixing)
			double link = 0.0 / 0.0;
			if (doGF)
			{
				auto rot = gaugeFix(m, 1.0e-11, 10000);
				link = avgLink(m, rot);
			}

			fmt::print("beta = {} / {}, <loop4> = {}, <link> = {}\n", beta,
			           beta2, loop4, link);

			// plot measurements
			pBeta.back().push_back(beta);
			pAction.back().push_back(loop4);
			pLink.back().push_back(link);

			plotAction.clear();
			for (size_t i = 0; i < pBeta.size(); ++i)
				plotAction.plotData(
				    pBeta[i], pAction[i],
				    fmt::format("<action> (b2 = {})", beta2s[i]));

			if (doGF)
			{
				plotLink.clear();
				for (size_t i = 0; i < pBeta.size(); ++i)
					plotLink.plotData(
					    pBeta[i], pLink[i],
					    fmt::format("<link> (b2 = {})", beta2s[i]));
			}
		}
	}
}
