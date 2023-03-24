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
	std::string start = "random";
	double epsilon = 1.0;
	int substeps = 8;
	int count = 100;
	std::optional<std::string> seed = {};
	int precision = 2; // 1=float, 2=double

	// others
	bool doPlot = false;
	std::string filename = "";
	bool allow_overwrite = false;

	CLI::App app{"Simulate pure gauge theory."};

	// physics options
	app.add_option("--group", group, "gauge group (u1, su2, su3=default)");
	app.add_option("--geom", geom, "geometry of the lattice");
	app.add_option("--beta", beta, "(inverse) coupling constant");

	// simulation options
	app.add_option("--start", start, "starting configuration (default=random)");
	app.add_option("--seed", seed, "seed for the rng (default=random)");
	app.add_option("--count", count, "number of Markov steps");
	app.add_option("--scheme", scheme, "integrator (2lf, 2mn, 4mn)");
	app.add_option("--epsilon", epsilon, "steps size for the HMC update");
	app.add_option("--substeps", substeps, "subdivison of delta");
	app.add_option("--precision", precision,
	               "FP precision (1=single, 2=double=default)");

	// misc
	app.add_flag("--plot", doPlot, "do some plots summarizing the trajectory");
	app.add_option("--filename", filename,
	               "hdf5 file to store configs (and metadata) in");
	app.add_flag("--force", allow_overwrite, "allow overwriting output file");

	CLI11_PARSE(app, argc, argv);

	if (!seed)
		seed = fmt::format("{}", std::random_device()());
	auto deltas = makeDeltas(scheme, epsilon, substeps);

	std::optional<util::Gnuplot> plot;
	if (doPlot)
		plot.emplace();
	util::Hdf5File file;
	if (filename != "")
	{
		file = util::Hdf5File::create(filename, allow_overwrite);

		// physical parameters
		file.set_attribute("group", group);
		file.set_attribute("beta", beta);
		file.set_attribute("geometry", geom);

		// simulation parameters
		file.set_attribute("hmc_scheme", scheme);
		file.set_attribute("hmc_epsilon", epsilon);
		file.set_attribute("hmc_substeps", substeps);
		file.set_attribute("hmc_metropolis", 1);
		file.set_attribute("markov_seed", seed);
		file.set_attribute("markov_start", start);
		file.set_attribute("markov_discard", 0);
		file.set_attribute("markov_count", count);
		file.set_attribute("markov_spacing", 1);

		file.make_group("/configs");
	}

	dispatchByGroup(
	    [&]<typename G>() {
		    auto g = Grid(Coordinate(geom.begin(), geom.end()));
		    auto hmc = Hmc<G>(g);
		    hmc.rng.seed(util::sha3<256>(seed.value()));

		    if (start == "random")
			    hmc.randomizeGaugeField();
		    else
			    hmc.U = readConfig<G>(start, &g);

		    for (size_t iter : util::ProgressRange(count))
		    {
			    hmc.runHmcUpdate(beta, deltas);

			    if ((iter + 1) % (100 / substeps) == 0 && plot)
			    {
				    plot->clear();
				    plot->plotData(std::span<double>(hmc.plaq_history)
				                       .subspan(hmc.plaq_history.size() / 10));
			    }

			    /*if (file)
			    {
			        std::string name = fmt::format("/configs/{}", iter + 1);
			        writeToFile(file, name, hmc.U);
			    }*/
		    }

		    if (plot)
		    {
			    plot->clear();
			    plot->plotData(hmc.plaq_history);

			    // util::Gnuplot().plotData(hmc.time_history);
		    }

		    hmc.print_summary();

		    if (file)
		    {
			    file.write_data("plaq_history", hmc.plaq_history);
			    file.write_data("accept_history", hmc.accept_history);
			    file.write_data("deltaH_history", hmc.deltaH_history);
			    file.write_data("time_history", hmc.time_history);
		    }
	    },
	    group, precision);

	swTotal.stop();

	fmt::print("========== performance summary ==========\n");
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
