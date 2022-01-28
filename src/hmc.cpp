#include "lattice/hmc.h"
#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "lattice/gauge.h"
#include "util/hash.h"

using namespace mesh;

int main(int argc, char **argv)
{
	util::Stopwatch swTotal;
	swTotal.start();

	// physics parameters
	std::string group = "su3";
	std::vector<int32_t> geom = {6, 6, 6, 6};
	double beta = 6.0;

	// simulation parameters
	std::string scheme = "4mn";
	double epsilon = 1.0;
	int substeps = 8;
	int count = 100;
	std::optional<std::string> seed = {};
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

	if (!seed)
		seed = fmt::format("{}", std::random_device()());
	auto deltas = makeDeltas(scheme, epsilon, substeps);

	std::optional<util::Gnuplot> plot;
	if (doPlot)
		plot.emplace();

	dispatchByGroup(
	    [&]<typename vG>() {
		    auto const &g = Grid::make(Coordinate(geom.begin(), geom.end()),
		                               TensorTraits<vG>::simdWidth);
		    auto hmc = Hmc<vG>(g);
		    hmc.rng.seed(util::sha3<256>(seed.value()));
		    hmc.randomizeGaugeField();
		    util::Stopwatch sw;
		    sw.start();
		    for (size_t iter : util::ProgressRange(count))
		    {
			    hmc.runHmcUpdate(beta, deltas);

			    if ((iter + 1) % (100 / substeps) == 0 && plot)
			    {
				    plot->clear();
				    plot->plotData(util::span<double>(hmc.plaq_history)
				                       .slice(hmc.plaq_history.size() / 10,
				                              hmc.plaq_history.size()));
			    }
		    }
		    sw.stop();
		    fmt::print("time per substep = {:.3f}\n",
		               sw.secs() / (count * substeps));

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

	swTotal.stop();

	fmt::print("lattice allocs: {}\n", latticeAllocCount);

	fmt::print("time in cshift:    {:#6.2f} s ({:#4.1f} %)\n", swCshift.secs(),
	           100. * swCshift.secs() / swTotal.secs());
	fmt::print("time in staples:   {:#6.2f} s ({:#4.1f} %)\n", swStaples.secs(),
	           100. * swStaples.secs() / swTotal.secs());
	fmt::print("time in rng:       {:#6.2f} s ({:#4.1f} %)\n", swRandom.secs(),
	           100. * swRandom.secs() / swTotal.secs());
	fmt::print("time in plaquette: {:#6.2f} s ({:#4.1f} %)\n",
	           swPlaquette.secs(), 100. * swPlaquette.secs() / swTotal.secs());
	fmt::print("time in reunitize: {:#6.2f} s ({:#4.1f} %)\n",
	           swReunitize.secs(), 100. * swReunitize.secs() / swTotal.secs());
	fmt::print("time in exp:       {:#6.2f} s ({:#4.1f} %)\n", swExp.secs(),
	           100. * swExp.secs() / swTotal.secs());
	fmt::print("total:             {:#6.2f} s\n", swTotal.secs());
}
