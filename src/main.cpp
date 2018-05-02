#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

#include "gnuplot.h"
#include "mesh.h"
#include "su2.h"
#include "z2.h"

#include "boost/program_options.hpp"
namespace po = boost::program_options;

int main(int argc, char **argv)
{
	po::options_description desc("Allowed options");
	desc.add_options()("help", "this help message")(
	    "n", po::value<int>()->default_value(8), "lattice size")(
	    "betaMin", po::value<double>()->default_value(0.0),
	    "inverse temperature")("betaMax",
	                           po::value<double>()->default_value(1.0), "")(
	    "warm", po::value<int>()->default_value(20),
	    "number of warmup sweeps")("meas", po::value<int>()->default_value(20),
	                               "number of measurment sweeps");

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
	auto nWarm = vm["warm"].as<int>();
	auto nMeas = vm["meas"].as<int>();

	auto plot = Gnuplot();
	plot.setRangeX(betaMin, betaMax);
	plot.setRangeY(0, 1);

	std::vector<double> xs, ys;

	for (int i = 0; i < 50; ++i)
	{
		double beta = betaMin + i * (betaMax - betaMin) / 49;

		auto m = Mesh<SU2>(Topology::lattice4D(n));
		// auto m = Mesh<Z2>(Topology::lattice4D(n));
		for (int i = 0; i < nWarm; ++i)
			m.thermalize(beta);

		double loop4 = 0;
		for (int i = 0; i < nMeas; ++i)
		{
			m.thermalize(beta);
			loop4 += m.loop4();
		}
		loop4 /= nMeas;

		xs.push_back(beta);
		ys.push_back(loop4);
		plot.clear();
		plot.plotData(xs, ys);
		std::cout << "beta = " << beta << ", <loop4> = " << loop4 << std::endl;
	}
}
