#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "lattice/gauge.h"
#include "util/gnuplot.h"
#include "util/linalg.h"
#include "util/progressbar.h"
#include "util/random.h"
#include "util/simd.h"

using namespace mesh;
using namespace mesh::QCD;

int main(int argc, char **argv)
{
	std::array<size_t, 4> geom = {8, 8, 8, 8};
	int count = 100;
	double beta = 6.0;
	int seed = -1;
	double delta = 0.1;
	bool doPlot = false;

	CLI::App app{"Simulate SU(3) pure gauge theory"};

	// physics options
	app.add_option("--geom", geom, "geometry of the lattice");
	app.add_option("--beta", beta, "(inverse) coupling constant");

	// simulation options
	app.add_option("--seed", seed, "seed for the rng (default=random)");
	app.add_option("--count", count, "number of Markov steps");
	app.add_option("--delta", delta, "delta-t for the HMC update");

	// misc
	app.add_flag("--plot", doPlot, "do some plots summarizing the trajectory");
	CLI11_PARSE(app, argc, argv);

	if (seed == -1)
		seed = std::random_device()();
	auto rng = util::xoshiro256(seed);

	auto const &g = Grid::make(Coordinate(geom.begin(), geom.end()), 4);
	auto U = randomGaugeField(g, rng);
	auto vol = g.size();

	std::vector<double> plaqHistory;
	for (size_t iter : util::ProgressRange(count))
	{
		// NOTE on conventions:
		//     * H = S(U) + 1/2 P^i P^i = S(U) - tr(P*P) = S(U) + norm2(P)
		//     * U' = P, P' = -S'(U)
		auto P = randomAlgebraField(g, rng);

		double H_old = wilsonAction(U, beta);
		for (int mu = 0; mu < Nd; ++mu)
			H_old += norm2(P[mu]);

		GaugeField U_new;
		for (int mu = 0; mu < Nd; ++mu)
			U_new[mu] = exp(P[mu] * (0.5 * delta)) * U[mu];
		for (int mu = 0; mu < Nd; ++mu)
			P[mu] -= wilsonDeriv(U_new, mu, beta) * delta;
		for (int mu = 0; mu < Nd; ++mu)
			U_new[mu] = exp(P[mu] * (0.5 * delta)) * U_new[mu];
		double H_new = wilsonAction(U_new, beta);
		for (int mu = 0; mu < Nd; ++mu)
			H_new += norm2(P[mu]);

		if (rng.uniform() < exp(H_old - H_new))
			std::swap(U, U_new);

		plaqHistory.push_back(plaquette(U));
	}

	fmt::print("plaquette = {} +- {}\n", util::mean(plaqHistory),
	           sqrt(util::variance(plaqHistory) / plaqHistory.size()));

	if (doPlot)
		util::Gnuplot().plotData(plaqHistory);
}
