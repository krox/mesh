#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "lattice/gauge.h"
#include "lattice/hmc.h"

using namespace mesh;

int main(int argc, char **argv)
{
	// physics parameters
	std::string group = "su3";
	std::vector<int32_t> geom = {6, 6, 6, 6};
	double beta = 6.0;

	// simulation parameters
	std::string scheme = "4mn";
	double epsilon = 1.0;
	int substeps = 8;
	int count = 100;
	int seed = -1;
	int precision = 2; // 1=float, 2=double

	// others
	bool doPlot = false;

	CLI::App app{"Simulate pure gauge theory."};

	// physics options
	app.add_option("--group", group, "gauge group (u1, su2, su3=default)");
	app.add_option("--geom", geom, "geometry of the lattice");
	app.add_option("--beta", beta, "(inverse) coupling constant");

	// simulation options
	app.add_option("--seed", seed, "seed for the rng (default=random)");
	app.add_option("--count", count, "number of Markov steps");
	app.add_option("--scheme", scheme, "integrator (2lf, 2mn, 4mn)");
	app.add_option("--epsilon", epsilon, "steps size for the HMC update");
	app.add_option("--substeps", substeps, "subdivison of delta");
	app.add_option("--precision", precision,
	               "FP precision (1=single, 2=double=default)");

	// misc
	app.add_flag("--plot", doPlot, "do some plots summarizing the trajectory");
	CLI11_PARSE(app, argc, argv);

	auto deltas = makeDeltas(scheme, epsilon, substeps);

	std::optional<util::Gnuplot> plot;
	if (doPlot)
		plot.emplace();

	dispatchByGroup(
	    [&]<typename vG>() {
		    auto const &g = Grid::make(Coordinate(geom.begin(), geom.end()),
		                               TensorTraits<vG>::simdWidth);
		    auto hmc = Hmc<vG>(g);
		    hmc.rng.seed(seed == -1 ? std::random_device()() : seed);
		    hmc.randomizeGaugeField();

		    for (size_t iter : util::ProgressRange(count))
		    {
			    hmc.runHmcUpdate(beta, deltas);

			    if ((iter + 1) % (100 / substeps) == 0 && plot)
			    {
				    plot->clear();
				    plot->plotData(util::span(hmc.plaq_history)
				                       .slice(hmc.plaq_history.size() / 10,
				                              hmc.plaq_history.size()));
			    }
		    }

		    fmt::print("plaquette = {} +- {}\n", util::mean(hmc.plaq_history),
		               sqrt(util::variance(hmc.plaq_history) /
		                    hmc.plaq_history.size()));
		    fmt::print("acceptance = {:.2f}\n", util::mean(hmc.accept_history));
		    fmt::print("<exp(-dH)> = {:.4f}\n",
		               util::mean(hmc.deltaH_history,
		                          [](double x) { return exp(-x); }));

		    if (plot)
		    {
			    plot->clear();
			    plot->plotData(hmc.plaq_history);
		    }
	    },
	    group, precision);

	fmt::print("lattice allocs: {}\n", latticeAllocCount);
	fmt::print("time in cshift: {:.2f}\n", swCshift.secs());
}
