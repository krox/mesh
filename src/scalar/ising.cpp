#include "scalar/ising.h"

#include "mesh/topology.h"
#include "util/hash.h"
#include "util/hdf5.h"
#include "util/progressbar.h"
#include "util/random.h"
#include "util/unionfind.h"
#include <cassert>
#include <fmt/format.h>
#include <random>
#include <vector>

namespace {

// math notes:
//     * action in Gibbs space: S = -beta * sum_(a,b) s[a] * s[b]
//	     (our convention: unordered links appear only once in sum)
//     * For cluster algorithms, this leads to
//           p = 1 - exp(-2*beta)
//     * action in Cluster space: S =
//       probability of a config: q^c (1 - p)^(1-w(e))
//     * random cluster measure:
//           prod_e p^w(e) * (1-p)^(1-w(e)) q^c(w)
//	     where
//	       w(e) = 1 indicates a bond, 0 the absence
//	       c(w) = number of components
//         for Ising, set q=2 and p=1-e^-beta
//     * action for cluster-ising:
//       S = - ln(exp(2*beta)-1) * #links - ln(2) * #components
//     * Gibbs -> Cluster: create links between sites with probability p
//       only between neighbouring sites with equal values
//     * Cluster -> Gibbs: assign each cluster an independent random value
//     * Swandson-Wang algorithm: Just alternate between between Cluster and
//       Gibbs. This does not suffer from the slowing down close to criticality
//       as basic single-link heatbath does.
//     * Wolff algorithm: similar to Swandson-Wang, but instead of finding all
//       clusters and flipping each with p=0.5, it finds a flips only a single
//       cluster with p=1.0 (allegedly this improves autocorrelation)

double isingAction(std::vector<int8_t> const &field, Topology const &top,
                   double beta)
{
	double sum = 0.0;
	for (auto const &link : top.links)
		sum += field[link.from] * field[link.to];
	return -beta * sum;
}

double isingMagnetization(std::vector<int8_t> const &field)
{
	double sum = 0.0;
	for (auto s : field)
		sum += s;
	return sum;
}

// run a single heat-bath sweep inplace
void heatBathSweep(std::vector<int8_t> &field, Topology const &top, double beta,
                   util::xoshiro256 &rng)
{
	// TODO: updating in some checkerboard order would be more natural
	for (int i = 0; i < top.nSites(); ++i)
	{
		// distribution of a site is determined by the sum of its neighbors
		double rho = 0;
		for (auto const &link : top.graph[i])
			rho += field[link.to];
		rho *= beta;

		// double p = exp(rho) / (exp(rho) + exp(-rho));
		double p = 1.0 / (1 + exp(-2 * rho));

		// make sure this update rule is 'monotone' (important for Propp-Wilson)
		field[i] = rng.uniform() < p ? 1.0 : -1.0;
	}
}

// opens hdf5 file, writes parameters, does nothing if filename==""
util::DataFile makeFile(IsingParams const &params, Topology const &top)
{
	if (params.filename == "")
		return {};

	auto file =
	    util::DataFile::create(params.filename, params.overwrite_existing);

	auto links = std::vector<int>(2 * top.nLinks());
	for (int i = 0; i < top.nLinks(); ++i)
	{
		links[2 * i] = top.links[i].from;
		links[2 * i + 1] = top.links[i].to;
	}

	// physical parameters
	file.setAttribute("beta", params.beta);
	file.setAttribute("geometry", params.geom);
	file.createData("topology", {(hsize_t)top.nLinks(), 2}, H5T_NATIVE_INT)
	    .write(links);

	// simulation parameters
	file.setAttribute("markov_count", params.count);
	file.setAttribute("markov_discard", params.discard);
	file.setAttribute("markov_spacing", params.spacing);

	file.makeGroup("/configs");
	return file;
}

// write configuration, does nothing if file is not valid
void writeConfig(util::DataFile &file, std::vector<int8_t> const &field,
                 std::vector<int> const &geom, int id)
{
	if (!file)
		return;

	// Possible values of the spins are 1,-1. There is no clean way in hdf5
	// to encode that in a single bit, and we dont want to deviate from
	// conventions in the analysis code. Therefore we take 2 bits.
	static hid_t hdf5type = 0;
	if (hdf5type == 0)
	{
		hdf5type = H5Tcopy(H5T_NATIVE_INT8);
		H5Tset_precision(hdf5type, 2);
	}

	std::vector<hsize_t> shape;
	for (int d : geom)
		shape.push_back(d);

	std::string name = fmt::format("/configs/{}", id);
	file.createData(name, shape, hdf5type).write(field);
}

} // namespace

IsingResults runHeatBath(const IsingParams &params)
{
	// state of the simulation
	auto top = Topology::lattice(params.geom);
	auto field = std::vector<int8_t>(top.nSites());
	util::xoshiro256 rng(util::fnv1a(params.seed));

	// results
	auto file = makeFile(params, top);
	IsingResults res;

	for (auto &x : field)
		x = -1;

	auto pb =
	    util::ProgressBar((params.count + params.discard) * params.spacing);
	for (int iter = -params.discard * params.spacing;
	     iter < params.count * params.spacing; ++iter, ++pb)
	{
		pb.show();

		// one sweep of single-site heat bath
		heatBathSweep(field, top, params.beta, rng);

		// measure observables
		if (iter >= 0)
		{
			res.actionHistory.push_back(isingAction(field, top, params.beta));
			res.magnetizationHistory.push_back(isingMagnetization(field));
		}

		// write config to file
		if (iter >= 0 && (iter + 1) % params.spacing == 0)
			writeConfig(file, field, params.geom, iter + 1);
	}
	pb.finish();

	if (file)
	{
		file.writeData("action_history", res.actionHistory);
		file.writeData("magnetization_history", res.magnetizationHistory);
	}
	return res;
}

IsingResults runSwendsenWang(const IsingParams &params)
{
	// state of the simulation
	auto top = Topology::lattice(params.geom);
	auto field = std::vector<int8_t>(top.nSites());
	util::xoshiro256 rng(util::fnv1a(params.seed));

	// stuff needed specifically for Swendsen-Wang
	double p = 1.0 - exp(-2 * params.beta);
	util::UnionFind uf(top.nSites());

	// results
	auto file = makeFile(params, top);
	IsingResults res;

	for (auto &x : field)
		x = -1;

	auto pb =
	    util::ProgressBar((params.count + params.discard) * params.spacing);
	for (int iter = -params.discard * params.spacing;
	     iter < params.count * params.spacing; ++iter, ++pb)
	{
		pb.show();

		double susceptibility = 0.0;

		uf.clear();
		for (auto &link : top.links)
			if (field[link.from] == field[link.to] && rng.uniform() < p)
				uf.join(link.from, link.to);
		for (int i = 0; i < top.nSites(); ++i)
			if (i == uf.root(i))
			{
				field[i] = rng.uniform() < 0.5 ? 1 : -1;

				// do some cluster-based measurements
				auto size = (double)uf.compSize(i);
				susceptibility += size * size;
			}
		for (int i = 0; i < top.nSites(); ++i)
			field[i] = field[uf.root(i)];

		// measure observables
		if (iter >= 0)
		{
			res.actionHistory.push_back(isingAction(field, top, params.beta));
			res.magnetizationHistory.push_back(isingMagnetization(field));
			susceptibility /= top.nSites();
			res.susceptibilityHistory.push_back(susceptibility);
		}

		// write config to file
		if (iter >= 0 && (iter + 1) % params.spacing == 0)
			writeConfig(file, field, params.geom, iter + 1);
	}
	pb.finish();

	if (file)
	{
		file.writeData("action_history", res.actionHistory);
		file.writeData("magnetization_history", res.magnetizationHistory);
		file.writeData("susceptibility_history", res.susceptibilityHistory);
	}
	return res;
}

IsingResults runProppWilson(const IsingParams &params)
{
	// state of the simulation
	auto top = Topology::lattice(params.geom);
	auto fieldPlus = std::vector<int8_t>(top.nSites());
	auto fieldMinus = std::vector<int8_t>(top.nSites());
	auto rng_master = util::Blake3(params.seed);

	// results
	util::DataFile file = makeFile(params, top);
	IsingResults res;

	// The Propp-Wilson algorithm is exact (no thermalization/autocorrelation).
	// Therefore discarding any configs is pointless
	assert(params.discard == 0 && params.spacing == 1);

	std::vector<double> T_history;

	auto pb =
	    util::ProgressBar((params.count + params.discard) * params.spacing);
	for (int iter = -params.discard * params.spacing;
	     iter < params.count * params.spacing; ++iter, ++pb)
	{
		pb.show();

		std::vector<util::xoshiro256> rngs;

		size_t T = 1;
		for (;; T *= 2)
		{
			while (rngs.size() < T)
				rngs.push_back(util::xoshiro256(rng_master()));

			// run markov from time -T to 0, starting from
			// all-plus and all-minus states
			for (auto &x : fieldPlus)
				x = 1;
			for (size_t i = 0; i < T; ++i)
			{
				auto rng = rngs[T - i]; // important: copy the RNG
				heatBathSweep(fieldPlus, top, params.beta, rng);
			}
			for (auto &x : fieldMinus)
				x = -1;
			for (size_t i = 0; i < T; ++i)
			{
				auto rng = rngs[T - i]; // important: copy the RNG
				heatBathSweep(fieldMinus, top, params.beta, rng);
			}

			// if the two results are equal, we are done
			bool done = true;
			for (size_t i = 0; i < fieldPlus.size(); ++i)
				if (fieldPlus[i] != fieldMinus[i])
				{
					done = false;
					break;
				}

			if (done)
				break;
			if (T >= 100000)
			{
				fmt::print("ERROR: Propp-Wilson did not find a solution with "
				           "100k steps. aborting.\n");
				exit(-1);
			}
		}

		fmt::print("found solution going back {} time-steps\n", T);
		T_history.push_back(T);

		// measure observables
		if (iter >= 0)
		{
			res.actionHistory.push_back(
			    isingAction(fieldMinus, top, params.beta));
			res.magnetizationHistory.push_back(isingMagnetization(fieldMinus));
		}

		// write config to file
		if (iter >= 0 && (iter + 1) % params.spacing == 0)
			writeConfig(file, fieldMinus, params.geom, iter + 1);
	}
	pb.finish();

	if (params.filename != "")
	{
		file.writeData("action_history", res.actionHistory);
		file.writeData("magnetization_history", res.magnetizationHistory);
		file.writeData("T_history", T_history);
	}
	return res;
}
