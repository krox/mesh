#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "lattice/gauge.h"
#include "lattice/hmc.h"

using namespace mesh;

int main(int argc, char **argv)
{
	HmcParams params = {};

	CLI::App app{"Simulate SU(3) pure gauge theory"};

	// physics options
	app.add_option("--group", params.group,
	               "gauge group (u1, su2, su3=default)");
	app.add_option("--geom", params.geom, "geometry of the lattice");
	app.add_option("--beta", params.beta, "(inverse) coupling constant");

	// simulation options
	app.add_option("--seed", params.seed, "seed for the rng (default=random)");
	app.add_option("--count", params.count, "number of Markov steps");
	app.add_option("--scheme", params.scheme, "integrator (2lf, 2mn, 4mn)");
	app.add_option("--epsilon", params.epsilon,
	               "steps size for the HMC update");
	app.add_option("--substeps", params.substeps, "subdivison of delta");

	// misc
	app.add_flag("--plot", params.doPlot,
	             "do some plots summarizing the trajectory");
	CLI11_PARSE(app, argc, argv);

	runHmc(params);
}
