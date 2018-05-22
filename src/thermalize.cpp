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
	("beta", po::value<std::vector<double>>()->multitoken(), "inverse temperature")
	("sweeps", po::value<int>()->default_value(100), "number of sweeps")
	("gaugefix", "do gauge-fixing and measure average link")
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
	std::vector<double> betas;
	if (vm.count("beta"))
		betas = vm["beta"].as<std::vector<double>>();
	else
		betas.push_back(0.5);
	auto nSweeps = vm["sweeps"].as<int>();
	bool doGF = vm.count("gaugefix");

	Gnuplot plotAction, plotLink;
	plotAction.setRangeX(0, nSweeps);
	plotAction.setRangeY(0, 1.5);
	plotLink.setRangeX(0, nSweeps);
	plotLink.setRangeY(0, 1.5);

	std::vector<std::vector<double>> pSweep, pAction, pLink;

	rng_t rng{std::random_device()()};

	for (double beta : betas)
	{
		pSweep.emplace_back();
		pAction.emplace_back();
		pLink.emplace_back();

		auto m = Mesh<Z2>(Topology::lattice4D(n));
		m.initMixed(rng);
		auto ga = GaugeAction(m);

		for (int iter = 0; iter < nSweeps; ++iter)
		{
			// measure average action
			double loop4 = ga.loop4();

			// measure average link (requires gauge fixing)
			double link = 0.0 / 0.0;
			if (doGF)
			{
				auto rot = gaugeFix(m, 1.0e-11, 10000);
				link = avgLink(m, rot);
			}

			fmt::print("beta = {} / {}, <loop4> = {}, <link> = {}\n", beta, 0,
			           loop4, link);

			// plot measurements
			pSweep.back().push_back(iter);
			pAction.back().push_back(loop4);
			pLink.back().push_back(link);

			plotAction.clear();
			for (size_t i = 0; i < pSweep.size(); ++i)
				plotAction.plotData(pSweep[i], pAction[i],
				                    fmt::format("<action> (b = {})", betas[i]));

			if (doGF)
			{
				plotLink.clear();
				for (size_t i = 0; i < pSweep.size(); ++i)
					plotLink.plotData(pSweep[i], pLink[i],
					                  fmt::format("<link> (b = {})", betas[i]));
			}

			// one thermalization sweep
			ga.thermalize(rng, beta, 0);
		}
	}
}
