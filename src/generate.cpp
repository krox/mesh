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

struct Config
{
	std::string group;    // gauge group;
	int n;                // lattice size
	double beta, beta2;   // inverse coupling
	int count;            // number of configs to generate
	int nTherms, nSweeps; // number of sweeps between measurements
	uint64_t seed;        // seed for random number generator
};

struct Result
{
	std::vector<double> loop4;
	double accProb = 0.0 / 0.0;
};

template <typename G> Result runGroup(const Config &cnf)
{
	rng_t rng(cnf.seed);

	// initialize mesh
	auto m = Mesh<G>(Topology::lattice4D(cnf.n));
	m.initMixed(rng);
	auto ga = GaugeAction(m);
	G::clearStats();

	// thermalization
	for (int i = 0; i < cnf.nTherms; ++i)
		ga.thermalize(rng, cnf.beta, cnf.beta2);

	// measurements
	Result r;
	for (int i = 0; i < cnf.count; ++i)
	{
		for (int j = 0; j < cnf.nSweeps; ++j)
			ga.thermalize(rng, cnf.beta, cnf.beta2);
		r.loop4.push_back(ga.loop4());
	}
	r.accProb = G::accProb();
	return r;
}

Result run(const Config &cnf)
{
	if (cnf.group == "z2")
		return runGroup<Z2>(cnf);
	else if (cnf.group == "u1")
		return runGroup<U1>(cnf);
	else if (cnf.group == "su2")
		return runGroup<SU2>(cnf);
	else
	{
		fmt::print("ERROR: unknown gauge group '{}'\n", cnf.group);
		exit(-1);
	}
}

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
	("therms", po::value<int>()->default_value(0), "number of sweeps for thermalization")
	("sweeps", po::value<int>()->default_value(20), "number of sweeps between configs")
	("seed", po::value<uint64_t>()->default_value(std::random_device()()), "seed for random number generator (same for all betas)")
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

	Config cnf;
	cnf.group = vm["group"].as<std::string>();
	cnf.n = vm["n"].as<int>();
	cnf.count = vm["count"].as<int>();
	cnf.nTherms = vm["therms"].as<int>();
	cnf.nSweeps = vm["sweeps"].as<int>();
	cnf.seed = vm["seed"].as<uint64_t>();

	assert(vm.count("beta"));
	assert(vm.count("beta2"));
	auto betas = vm["beta"].as<std::vector<double>>();
	auto beta2s = vm["beta2"].as<std::vector<double>>();

	bool doPlot = vm.count("plot");

	fmt::print("╔═════════════╤═══════════╤══════╤═══════╗\n");
	fmt::print("║  coupling   │   plaq    │  acc │   ac  ║\n");
	fmt::print("╟─────────────┼───────────┼──────┼───────╢\n");

	for (double beta2 : beta2s)
	{
		for (double beta : betas)
		{
			cnf.beta = beta;
			cnf.beta2 = beta2;

			auto res = run(cnf);

			Autocorrelation ac;
			for (double x : res.loop4)
				ac.add(x);
			fmt::print("║ {:5.3f} {:5.3f} │ {:9.7f} │ {:4.2f} │ {:5.2f} ║\n",
			           beta, beta2, ac.mean(), res.accProb, ac.corr(1));

			if (doPlot)
			{
				Gnuplot().plotData(
				    res.loop4,
				    fmt::format("<action> (b={}, b2={})", beta, beta2));
				std::vector<double> acs;
				for (size_t i = 0; i < 20; ++i)
					acs.push_back(ac.corr(i));
				Gnuplot().plotData(
				    acs, fmt::format("auto-corr (b={}, b2={})", beta, beta2));
			}
		}
	}

	fmt::print("╚═════════════╧═══════════╧══════╧═══════╝\n");
}
