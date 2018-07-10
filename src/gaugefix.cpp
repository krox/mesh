#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

#include <fmt/format.h>

//#define LOG_GAUGEFIXING

#include "mesh/gauge_fixing.h"
#include "mesh/mesh.h"
#include "mesh/su2.h"
#include "mesh/u1.h"
#include "mesh/wilson.h"
#include "mesh/z2.h"
#include "util/gnuplot.h"
#include "util/io.h"

#include "boost/program_options.hpp"
namespace po = boost::program_options;

/** writes results directly into ensemble file */
template <typename G> void run(DataFile &file, double eps)
{
	// parameters of the ensemble
	auto group = file.getAttribute<std::string>("group");
	auto top = file.getAttribute<std::string>("topology");
	auto geom = file.getAttribute<std::vector<int>>("geometry");
	auto count = file.getAttribute<int>("markov_count");
	// assert(group == G::name());
	assert(top == "periodic4");

	auto m = Mesh<G>(Topology::lattice(geom));
	for (int i = 0; i < 1; ++i)
	{
		fmt::print("reading config {} / {} ...\n", i + 1, count);
		file.openData(fmt::format("configs/{}", i + 1)).read(m.rawLinks());
		GaugeFixing<G> fix(m);
		fix.verbose = true;
		fix.relax(eps, 100000);
	}
}

int main(int argc, char **argv)
{
	// clang-format off
	po::options_description desc("Allowed options");
	desc.add_options()
	("help", "this help message")
	("file", po::value<std::string>(), "file input/output (HDF5 format)")
	("eps", po::value<double>()->default_value(1.0e-11), "desired precision")
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

	auto filename = vm["file"].as<std::string>();
	auto eps = vm["eps"].as<double>();

	auto file = DataFile::open(filename);

	auto group = file.getAttribute<std::string>("group");

	if (group == "u1")
		run<U1>(file, eps);
	else if (group == "su2")
		run<SU2>(file, eps);
	else
		throw std::runtime_error("unknown group");
}
