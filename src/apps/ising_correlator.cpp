#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "lattice/fft.h"
#include "mesh/topology.h"
#include "scalar/ising.h"
#include "util/bit_vector.h"
#include "util/hdf5.h"
#include "util/random.h"
#include "util/unionfind.h"
#include <cassert>
#include <cmath>
#include <filesystem>
#include <random>
#include <vector>

using namespace mesh;

std::vector<double> analyze_basic(std::vector<int8_t> const &field, double beta,
                                  Topology const &top, util::xoshiro256 &rng,
                                  int samples)
{
	(void)beta;
	(void)rng;
	(void)samples;
	auto corr = Correlator(top.geom);

	auto c2pt = std::vector<double>(top.nSites());
	for (int i = 0; i < top.nSites(); ++i)
		corr.in()[i] = field[i];
	corr.run();
	for (int i = 0; i < top.nSites(); ++i)
		c2pt[i] = corr.out()[i].real() / top.nSites() / top.geom.back();
	return c2pt;
}

std::vector<double> analyze(std::vector<int8_t> const &field, double beta,
                            Topology const &top, util::xoshiro256 &rng,
                            int samples)
{
	// TODO:
	//     * only store a few (small) momenta
	//     * optimize (and profile first):
	//           - small clusters (at least size=1) dont need FFT
	//           - full complex FFT is overkill
	//           - allocation oerhead could be reduced
	assert((int)field.size() == top.nSites());

	auto uf = util::UnionFind(top.nSites());
	double p = 1.0 - std::exp(-2.0 * beta);
	auto c2pt = std::vector<double>(top.nSites(), 0.0);
	auto corr = Correlator(top.geom);

	for (int iter = 0; iter < samples; ++iter)
	{
		uf.clear();

		for (auto [a, b] : top.links)
			if (field[a] == field[b] && rng.uniform() < p)
				uf.join(a, b);
		auto comps = uf.components();

		for (int comp = 0; comp < (int)uf.nComp(); ++comp)
		{
			for (int i = 0; i < top.nSites(); ++i)
				corr.in()[i] = comps[i] == comp ? 1.0 : 0.0;
			corr.run();

			for (int i = 0; i < top.nSites(); ++i)
				c2pt[i] += corr.out()[i].real();
		}
	}

	for (auto &c : c2pt)
		c *= 1.0 / samples / top.nSites() / top.geom.back();
	return c2pt;
}

std::vector<double> analyze_binomial(std::vector<int8_t> const &field,
                                     double beta, Topology const &top,
                                     util::xoshiro256 &rng, int samples)
{
	assert(field.size() == (size_t)top.nSites());

	// potential bonds between neighbours of equal color
	std::vector<int> valid_links;
	valid_links.reserve(top.nLinks());
	for (int i = 0; i < top.nLinks(); ++i)
		if (field[top.links[i].first] == field[top.links[i].second])
			valid_links.push_back(i);

	auto visit_site = util::bit_vector(top.nSites());
	auto visit_link = util::bit_vector(top.nLinks());
	double p = 1 - std::exp(-2 * beta);
	auto c2pt = std::vector<double>(field.size(), 0.0);
	std::vector<int> stack;
	std::vector<double> weights =
	    util::binomial_distribution((int)valid_links.size(), p).pdf();
	for (int i = weights.size() - 2; i >= 0; --i)
		weights[i] += weights[i + 1];

	for (int iter = 0; iter < samples; ++iter)
	{
		std::shuffle(valid_links.begin(), valid_links.end(), rng);
		visit_site.clear();
		visit_link.clear();

		visit_site[0] = true;
		c2pt[0] += 1.0;

		int link_count = 0;
		for (int i : valid_links)
		{
			double weight = weights[++link_count];
			auto [a, b] = top.links[i];

			// both done -> do nothing
			if (visit_site[a] && visit_site[b])
				continue;

			// neither done -> mark for later
			if (!visit_site[a] && !visit_site[b])
			{
				visit_link[i] = true;
				continue;
			}

			// a visited, b not -> found something new!
			if (!visit_site[a])
				std::swap(a, b);
			assert(visit_site[a] && !visit_site[b]);

			assert(stack.empty());
			stack.push_back(b);
			visit_site[b] = true;
			while (!stack.empty())
			{
				auto x = stack.back();
				stack.pop_back();

				c2pt[x] += weight;
				for (auto hl : top.graph[x])
					if (!visit_site[hl.to] && visit_link[hl.link.i])
					{
						stack.push_back(hl.to);
						visit_site[hl.to] = true;
					}
			}
		}
	}

	for (auto &c : c2pt)
		c *= 1.0 / samples;
	return c2pt;
}

int main(int argc, char **argv)
{
	// parse program options

	bool overwrite_existing = false;
	std::string seed = "";
	std::string filename;
	int samples = 10;

	CLI::App app{"Compute 2-point correlator of Ising model with an improved "
	             "estimator based on random cluster model"};

	app.add_option("--filename", filename,
	               "hdf5 input (configs) and output (correlator)")
	    ->required();
	app.add_option("--samples", samples,
	               "number of randomized samples to take per config");
	app.add_flag("--force", overwrite_existing, "overwrite existing data");
	app.add_option(
	    "--seed", seed,
	    "seed for random number generator (default = empty = random)");

	CLI11_PARSE(app, argc, argv);

	if (seed.empty())
		seed = fmt::format("seed_ising_2pt_{}", std::random_device()());
	auto rng = util::xoshiro256(seed);

	// read parameters from file

	auto file = util::Hdf5File::open(filename, true);
	auto beta = file.get_attribute<double>("beta");
	auto count = file.get_attribute<int>("markov_count");
	// auto top = Topology(file.read_data<std::pair<int, int>>("topology"));
	auto top =
	    Topology::lattice(file.get_attribute<std::vector<int>>("geometry"));

	if (file.exists("c2pt_improved"))
	{
		if (overwrite_existing)
			file.remove("c2pt_improved");
		else
		{
			fmt::print("measurments exists already. skipping.");
			return 0;
		}
	}

	file.make_group("c2pt_improved");
	std::vector<hsize_t> shape;
	shape.push_back(count);
	for (auto s : top.geom)
		shape.push_back(s);
	auto dataset = file.create_data("c2pt_improved/full", shape);

	for (int i = 0; i < count; ++i)
	{
		auto field = file.read_data<int8_t>(fmt::format("configs/{}", i + 1));
		auto c2pt = analyze(field, beta, top, rng, samples);
		dataset.write(i, c2pt);
	}
}
