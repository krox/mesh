#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "lattice/gauge.h"
#include "lattice/hmc.h"
#include "util/hash.h"

using namespace mesh;

int main(int argc, char **argv)
{
	// physics parameters
	// (default is reasonable to produce a plot like [Gattringer, page 99])
	std::string group = "su3";
	std::vector<int32_t> geom = {4, 4, 4, 4};
	double beta_min = 0;
	double beta_max = 8;
	int beta_count = 30;

	// simulation parameters
	std::string scheme = "4mn";
	double epsilon = 2.0;
	int substeps = 8;
	int count = 50;
	std::optional<std::string> seed = {};
	int precision = 2; // 1=float, 2=double

	// others
	bool doPlot = false;

	CLI::App app{"Simulate pure gauge theory for a range of couplings."};

	// physics options
	app.add_option("--group", group, "gauge group (u1, su2, su3=default)");
	app.add_option("--geom", geom, "geometry of the lattice");
	app.add_option("--beta-min", beta_min);
	app.add_option("--beta-max", beta_max);
	app.add_option("--beta-count", beta_count);

	// simulation options
	app.add_option("--seed", seed, "seed for the rng (default=random)");
	app.add_option("--count", count, "number of Markov steps per beta");
	app.add_option("--scheme", scheme, "integrator (2lf, 2mn, 4mn)");
	app.add_option("--epsilon", epsilon, "steps size for the HMC update");
	app.add_option("--substeps", substeps, "subdivison of delta");
	app.add_option("--precision", precision,
	               "FP precision (1=single, 2=double=default)");

	// misc
	app.add_flag("--plot", doPlot, "do some plots in the end");
	CLI11_PARSE(app, argc, argv);

	if (!seed)
		seed = fmt::format("{}", std::random_device()());
	auto deltas = makeDeltas(scheme, epsilon, substeps);

	std::optional<util::Gnuplot> plot;
	if (doPlot)
		plot.emplace();
	std::vector<double> xs, ys;
	dispatchByGroup(
	    [&]<typename vG>() {
		    auto const &g = Grid(Coordinate(geom.begin(), geom.end()));
		    auto hmc = Hmc<vG>(g);
		    hmc.rng.seed(util::sha3<256>(seed.value()));
		    hmc.randomizeGaugeField();

		    for (int iter = 0; iter < beta_count; ++iter)
		    {
			    double beta =
			        beta_min + (beta_max - beta_min) / (beta_count - 1) * iter;

			    hmc.reset_observables();
			    for (int i = 0; i < count; ++i)
				    hmc.runHmcUpdate(beta, deltas);

			    auto plaq =
			        util::mean(std::span(hmc.plaq_history)
			                       .subspan(hmc.plaq_history.size() / 10,
			                                hmc.plaq_history.size()));
			    fmt::print("beta = {:.3f}, acc = {:.3f}, plaq = {:.3f}\n", beta,
			               util::mean(hmc.accept_history), plaq);

			    xs.push_back(beta);
			    ys.push_back(plaq);
			    if (plot)
			    {
				    plot->clear();
				    plot->plotData(xs, ys);

				    // some known expansions
				    if (group == "su3")
					    plot->plotFunction("x/18", 0.0, 3.0, "strong coupling");
			    }
		    }
	    },
	    group, precision);

	fmt::print("lattice allocs: {}\n", latticeAllocCount);
	fmt::print("time in cshift: {:.2f}\n", swCshift.secs());
}
