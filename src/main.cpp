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

typedef xoroshiro128plus rng_t;

struct Result
{
	double loop4 = 0.0 / 0.0;
	double accProb = 0.0 / 0.0;
};

template <typename G>
Result run(int n, double beta, double beta2, double nSweeps, rng_t &rng)
{
	// initialize mesh
	auto m = Mesh<G>(Topology::lattice4D(n));
	m.initMixed(rng);
	auto ga = GaugeAction(m);
	G::clearStats();

	// thermalization
	for (int i = 0; i < nSweeps; ++i)
		ga.thermalize(rng, beta, beta2);

	// measurements
	Result r;
	r.loop4 = ga.loop4();
	r.accProb = G::accProb();
	return r;
}

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
	("sweeps", po::value<int>()->default_value(20), "number of sweeps between measurements")
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

	auto group = vm["group"].as<std::string>();
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

	Gnuplot plotAction, plotLink;
	plotAction.setRangeX(betaMin, betaMax);
	plotAction.setRangeY(0, 1);

	std::vector<std::vector<double>> pBeta;
	std::vector<std::vector<double>> pAction;

	rng_t rng{std::random_device()()};

	fmt::print("╔═════════════╤═══════════╤══════╗\n");
	fmt::print("║  coupling   │   plaq    │ acc  ║\n");
	fmt::print("╟─────────────┼───────────┼──────╢\n");

	for (double beta2 : beta2s)
	{
		pBeta.emplace_back();
		pAction.emplace_back();

		for (int i = 0; i < nSteps; ++i)
		{
			double beta = betaMin + i * (betaMax - betaMin) / (nSteps - 1);

			Result res;
			if (group == "z2")
				res = run<Z2>(n, beta, beta2, nSweeps, rng);
			else if (group == "u1")
				res = run<U1>(n, beta, beta2, nSweeps, rng);
			else if (group == "su2")
				res = run<SU2>(n, beta, beta2, nSweeps, rng);
			else
				assert(false);

			fmt::print("║ {:5.3f} {:5.3f} │ {:9.7f} │ {:4.2f} ║\n", beta, beta2,
			           res.loop4, res.accProb);

			// plot measurements
			pBeta.back().push_back(beta);
			pAction.back().push_back(res.loop4);

			plotAction.clear();
			for (size_t i = 0; i < pBeta.size(); ++i)
				plotAction.plotData(
				    pBeta[i], pAction[i],
				    fmt::format("<action> (b2 = {})", beta2s[i]));
		}
	}

	fmt::print("╚═════════════╧═══════════╧══════╝\n");
}
