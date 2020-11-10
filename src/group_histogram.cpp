#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "groups/su2.h"
#include "groups/u1.h"
#include "groups/z2.h"
#include "util/gnuplot.h"
#include "util/random.h"
#include "util/stats.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

template <typename G> void run(double a, double a2, int n)
{
	const int binCount = 50;
	const int intCount = 100 * binCount;

	// generate the histogram
	rng_t rng{std::random_device()()};
	util::Histogram hist(-1, 1, binCount);
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
	util::Gnuplot p;
	p.setLogScaleY();
	p.setRangeX(-1.05, 1.05);
	p.plotHistogram(hist);
	p.plotFunction([&](double x) { return c * G::dist(a, a2, x); }, -1, 1);
}

int main(int argc, char **argv)
{
	std::string group = "u1";
	double alpha = 1.0;
	double alpha2 = 0.0;
	int n = 1000000;

	CLI::App app{"Generate elements of a gauge-group and plot a histogram "
	             "compared to the expected distribution."};
	// physics options
	app.add_option("--group,-g", group, "gauge group (u1,su2)");
	app.add_option("--alpha", alpha, "linear coefficient");
	app.add_option("--alpha2", alpha2, "quadratic coefficient");
	app.add_option("--count,-n", n, "number of samples to generate");

	CLI11_PARSE(app, argc, argv);

	if (group == "u1")
		run<U1>(alpha, alpha2, n);
	else if (group == "su2")
		run<SU2>(alpha, alpha2, n);
	else
		throw std::runtime_error("unknown gauge group");
}
