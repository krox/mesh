#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

#include "gauge_action.h"
#include "gnuplot.h"
#include "mesh.h"
#include "random.h"
#include "su2.h"
#include "u1.h"
#include "z2.h"

#include "boost/program_options.hpp"
namespace po = boost::program_options;

int main(int argc, char **argv)
{
	// clang-format off
	po::options_description desc("Allowed options");
	desc.add_options()
	("help", "this help message")
	("n", po::value<int>()->default_value(8), "lattice size")
	("betaMin", po::value<double>()->default_value(0.0), "inverse temperature")
	("betaMax", po::value<double>()->default_value(1.0), "")
	("beta2", po::value<std::vector<double>>()->multitoken(), "secondary (usually adjoint) coupling");
	("warm", po::value<int>()->default_value(20), "number of warmup sweeps")
	("meas", po::value<int>()->default_value(20), "number of measurment sweeps");
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
	auto beta2s = vm["beta2"].as<std::vector<double>>();
	if (beta2s.empty())
		beta2s.push_back(0.0);
	auto nWarm = vm["warm"].as<int>();
	auto nMeas = vm["meas"].as<int>();

	auto plot = Gnuplot();
	plot.setRangeX(betaMin, betaMax);
	plot.setRangeY(0, 1);

	std::vector<std::vector<double>> xs, ys;

	xoroshiro128plus rng{std::random_device()()};

	for (double beta2 : beta2s)
	{
		xs.emplace_back();
		ys.emplace_back();

		for (int i = 0; i < 50; ++i)
		{
			double beta = betaMin + i * (betaMax - betaMin) / 49;

			auto m = Mesh<U1>(Topology::lattice4D(n));
			auto ga = GaugeAction(m);

			for (int i = 0; i < nWarm; ++i)
				ga.thermalize(rng, beta, beta2);

			double loop4 = 0;
			for (int i = 0; i < nMeas; ++i)
			{
				ga.thermalize(rng, beta, beta2);
				loop4 += ga.loop4();
			}
			loop4 /= nMeas;

			xs.back().push_back(beta);
			ys.back().push_back(loop4);

			std::cout << "beta = " << beta << ", <loop4> = " << loop4
			          << std::endl;

			plot.clear();
			for (size_t i = 0; i < xs.size(); ++i)
			{
				plot.plotData(xs[i], ys[i]);
			}
		}
	}
}
