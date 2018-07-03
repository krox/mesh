#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

#include <fmt/format.h>

#include "mesh/gauge_action.h"
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
template <typename G> void computeWilson(DataFile &file, int maxN)
{
	// parameters of the ensemble
	auto group = file.getAttribute<std::string>("group");
	auto top = file.getAttribute<std::string>("topology");
	auto geom = file.getAttribute<std::vector<int>>("geometry");
	auto count = file.getAttribute<int>("markov_count");
	// assert(group == G::name());
	assert(top == "periodic4");

	auto m = Mesh<G>(Topology::lattice(geom));
	auto buf = std::vector<double>(count * (maxN + 1) * (geom[3] + 1), 1.0);
	auto result = Lattice<double, 3>(buf, {count, maxN + 1, geom[3] + 1});
	for (int i = 0; i < count; ++i)
	{
		fmt::print("reading config {} / {} ...\n", i + 1, count);
		file.openData(fmt::format("configs/{}", i + 1)).read(m.rawLinks());
		for (auto [n, t, s] : wilson(m, maxN))
			result({i, n, t}) = s;
	}

	fmt::print("writing result...\n");
	file.createData("wilson", {(unsigned)count, (unsigned)maxN + 1,
	                           (unsigned)geom[3] + 1})
	    .write(buf);
	fmt::print("done\n");
}

int main(int argc, char **argv)
{
	// clang-format off
	po::options_description desc("Allowed options");
	desc.add_options()
	("help", "this help message")
	("file", po::value<std::string>(), "file input/output (HDF5 format)")
	("maxN,n", po::value<int>()->default_value(-1), "maxN (default = full latt size)")
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
	auto n = vm["maxN"].as<int>();

	auto file = DataFile::open(filename);

	if (file.exists("wilson"))
	{
		fmt::print("file '{}' already done. skipping.", filename);
		return 0;
	}

	auto group = file.getAttribute<std::string>("group");
	if (n == -1)
		n = file.getAttribute<std::vector<int>>("geometry")[0];

	if (group == "z2")
		computeWilson<Z2>(file, n);
	else if (group == "u1")
		computeWilson<U1>(file, n);
	else if (group == "su2")
		computeWilson<SU2>(file, n);
	else
		throw std::runtime_error("unknown group");
}
