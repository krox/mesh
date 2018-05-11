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
#include "util/stats.h"

#include "boost/program_options.hpp"
namespace po = boost::program_options;

template <typename G> void run(double a, double a2, int n)
{
	const int binCount = 50;
	const int intCount = 100 * binCount;

	// generate the histogram
	xoroshiro128plus rng{std::random_device()()};
	histogram hist(-1, 1, binCount);
	for (int i = 0; i < n; ++i)
		hist.add(G::random(rng, a, a2).action());

	// poor man's integration (TODO: do something sensible here)
	double c = 0;
	for (int i = 1; i < intCount; ++i) // beware of singularities at +-1
		c += G::dist(a, a2, 2.0 * i / intCount - 1.0);
	c /= intCount - 1;

	// scale the distribution
	c = n / binCount / c;

	// plot everything
	Gnuplot p;
	p.setLogScaleY();
	p.setRangeX(-1.05, 1.05);
	p.plotHistogram(hist);
	p.plotFunction([&](double x) { return c * G::dist(a, a2, x); }, -1, 1);
}

int main(int argc, char **argv)
{
	// clang-format off
	po::options_description desc("Allowed options");
	desc.add_options()
	("help", "this help message")
	("group,g", po::value<std::string>()->default_value("u1"), "gauge group")
	("alpha", po::value<double>()->default_value(1.0), "linear coefficient")
	("alpha2", po::value<double>()->default_value(0.0), "quadratic coefficient")
	("n", po::value<int>()->default_value(1000000), "number of samples to take")
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

	auto g = vm["group"].as<std::string>();
	auto alpha = vm["alpha"].as<double>();
	auto alpha2 = vm["alpha2"].as<double>();
	auto n = vm["n"].as<int>();

	if (g == "u1")
		run<U1>(alpha, alpha2, n);
	/*else if (g == "su2")
	    run<SU2>(alpha, alpha2, n);*/
	else
		throw "unknown gauge group";
}
