#include "scalar/ising.h"

#include "mesh/topology.h"
#include "util/hash.h"
#include "util/hdf5.h"
#include "util/progressbar.h"
#include "util/random.h"
#include "util/stopwatch.h"
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
util::Hdf5File makeFile(IsingParams const &params, Topology const &top)
{
	if (params.filename == "")
		return {};

	auto file =
	    util::Hdf5File::create(params.filename, params.overwrite_existing);

	auto links = std::vector<int>(2 * top.nLinks());
	for (int i = 0; i < top.nLinks(); ++i)
	{
		links[2 * i] = top.links[i].from;
		links[2 * i + 1] = top.links[i].to;
	}

	// physical parameters
	file.set_attribute("beta", params.beta);
	file.set_attribute("geometry", params.geom);
	file.create_data("topology", {(hsize_t)top.nLinks(), 2}, H5T_NATIVE_INT)
	    .write(links);

	// simulation parameters
	file.set_attribute("markov_count", params.count);
	file.set_attribute("markov_discard", params.discard);
	file.set_attribute("markov_spacing", params.spacing);

	file.make_group("/configs");
	return file;
}

// write configuration, does nothing if file is not valid
void writeConfig(util::Hdf5File &file, std::vector<int8_t> const &field,
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
	file.create_data(name, shape, hdf5type).write(field);
}

} // namespace

IsingResults run_heat_bath(const IsingParams &params)
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
		file.write_data("action_history", res.actionHistory);
		file.write_data("magnetization_history", res.magnetizationHistory);
	}
	return res;
}

IsingResults run_exact_heat_bath(const IsingParams &params)
{
	// state of the simulation
	auto top = Topology::lattice(params.geom);
	auto fieldPlus = std::vector<int8_t>(top.nSites());
	auto fieldMinus = std::vector<int8_t>(top.nSites());
	util::Blake3 rng_master(params.seed);

	// results
	util::Hdf5File file = makeFile(params, top);
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

		// fmt::print("found solution going back {} time-steps\n", T);
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
		file.write_data("action_history", res.actionHistory);
		file.write_data("magnetization_history", res.magnetizationHistory);
		file.write_data("T_history", T_history);
	}
	return res;
}

IsingResults run_swendsen_wang(const IsingParams &params)
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
		file.write_data("action_history", res.actionHistory);
		file.write_data("magnetization_history", res.magnetizationHistory);
		file.write_data("susceptibility_history", res.susceptibilityHistory);
	}
	return res;
}
namespace {

class ExactSwendsenWang
{
	Topology top_;

	// no information is kept in these between calls to .run(), only the storage
	std::vector<int8_t> field_;
	std::vector<int8_t> bonds_;
	std::vector<int> stack_;
	std::vector<int> sites_;

	// make bonds out of the field
	void make_bonds(double beta, util::xoshiro256 &rng)
	{
		double p = 1.0 - exp(-2 * beta);

		for (int i = 0; i < top_.nLinks(); ++i)
		{
			bool coin = rng.uniform() <= p;
			auto a = field_[top_.links[i].from];
			auto b = field_[top_.links[i].to];
			if (coin && (a & b))
			{
				if (a < 3 && b < 3)
					bonds_[i] = 2; // definitely linked
				else
					bonds_[i] = 1; // maybe linked
			}
			else
				bonds_[i] = 0; // definitely not linked
		}
	}

	// make field out of the bonds
	void make_field(util::xoshiro256 &rng)
	{
		for (auto &x : field_)
			x = 0;

		// sites_ is a permutation of [0, nSites)
		// for (int root = 0; root < top.nSites(); ++root)
		for (int root : sites_)
		{
			int8_t root_color = rng.bernoulli() ? 1 : 2;
			// already part of a component -> skip
			// important: always sample the rng for consistent bounding chain
			if (field_[root])
				continue;

			// start a new component
			auto possible_colors = root_color;

			stack_.push_back(root);
			field_[root] = root_color;
			while (!stack_.empty())
			{
				int a = stack_.back();
				stack_.pop_back();

				for (auto &link : top_.graph[a])
				{
					if (bonds_[link.link.i] == 1)
					{
						possible_colors |= field_[link.to];
						if (possible_colors == 3)
							goto contaminated;
					}
					if (bonds_[link.link.i] == 2)
					{
						if (!field_[link.to])
						{
							field_[link.to] = root_color;
							stack_.push_back(link.to);
							continue;
						}
						// else must be in this component already
					}
				}
			}

		contaminated:
			//  possible link to different-colored previous component
			if (possible_colors != root_color)
			{
				field_[root] = possible_colors;
				stack_.push_back(root);
				while (!stack_.empty())
				{
					int a = stack_.back();
					stack_.pop_back();
					for (auto &link : top_.graph[a])
						if (bonds_[link.link.i] == 2 &&
						    field_[link.to] != possible_colors)
						{
							field_[link.to] = possible_colors;
							stack_.push_back(link.to);
						}
				}
			}
		}
	}

  public:
	// statistics gathered during generation
	std::vector<int> T_history;

	explicit ExactSwendsenWang(Topology top)
	    : top_(std::move(top)), field_(top_.nSites()), bonds_(top_.nLinks()),
	      sites_(top_.nSites())
	{}

	Topology const &topology() { return top_; }

	// create one new configuration
	std::vector<int8_t> const &run(double beta, util::xoshiro256 &rng)
	{
		// Changing the iteration order of the sites does not meaningfully
		// affect the Swendsen-Wang Markov chain itself, but it does have a
		// significant effect on the quality of the bounding chain, i.e., how
		// long it takes to detect complete coupling. It seems to be:
		//     * random order is better standard (lexicographic) ordering, but
		//     * changing the order in between steps is detrimental
		assert(sites_.size() == (size_t)top_.nSites());
		for (int i = 0; i < top_.nSites(); ++i)
			sites_[i] = i;
		std::shuffle(sites_.begin(), sites_.end(), rng);

		std::vector<util::xoshiro256> rngs;

		for (size_t T = 1; T < 10'000; T *= 2)
		{
			while (rngs.size() < T)
				rngs.push_back(rng.jump());

			// run markov from time -T to 0, starting from
			// all-plus and all-minus states
			for (auto &x : field_)
				x = 1 | 2;
			for (size_t i = 0; i < T; ++i)
			{
				auto local_rng = rngs[T - 1 - i];
				make_bonds(beta, local_rng);
				make_field(local_rng);
			}

			// if the two results are equal, we are done
			bool done = true;
			for (size_t i = 0; i < field_.size(); ++i)
				if (field_[i] == 3)
				{
					done = false;
					break;
				}

			if (done)
			{
				T_history.push_back(T);

				// transform 1/2 to 1/-1
				for (auto &x : field_)
					x = int8_t(-2 * x + 3);

				return field_;
			}
		}

		throw std::runtime_error("ERROR: Swendsen-Wang-Propp-Wilson did not "
		                         "find a solution within 10k steps. aborting.");
	}
};

} // namespace

// params.discard and params.spacing are ignored here
IsingResults run_exact_swendsen_wang(const IsingParams &params)
{
	// A single xoshiro256 instance would probably provide enough randomness
	// for the whole process. But as we use layered RNGs anyway, it is kinda
	// cool to use a CSRNG on the outermost level.
	auto rng = util::Blake3(params.seed);
	auto algo = ExactSwendsenWang(Topology::lattice(params.geom));

	// results
	util::Hdf5File file = makeFile(params, algo.topology());
	IsingResults res;

	auto pb = util::ProgressBar(params.count);
	for (int iter = 0; iter < params.count; ++iter, ++pb)
	{
		pb.show();

		auto chain_rng = util::xoshiro256(rng());
		auto &field = algo.run(params.beta, chain_rng);

		// measure observables
		if (iter >= 0)
		{
			res.actionHistory.push_back(
			    isingAction(field, algo.topology(), params.beta));
			res.magnetizationHistory.push_back(isingMagnetization(field));
		}

		// write config to file
		if (iter >= 0 && (iter + 1) % params.spacing == 0)
			writeConfig(file, field, params.geom, iter + 1);
	}
	pb.finish();

	if (params.filename != "")
	{
		file.write_data("action_history", res.actionHistory);
		file.write_data("magnetization_history", res.magnetizationHistory);
		file.write_data("T_history", algo.T_history);
	}

	return res;
}
