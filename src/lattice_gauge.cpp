#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "lattice/gauge.h"
#include "util/gnuplot.h"
#include "util/linalg.h"
#include "util/random.h"
using namespace mesh;
using namespace mesh::QCD;

int main(int argc, char **argv)
{
	std::array<size_t, 4> geom = {4, 4, 4, 4};
	double beta = 6.0;
	int seed = -1;

	CLI::App app{"Simulate SU(3) pure gauge theory"};

	// physics options
	app.add_option("--geom", geom, "geometry of the lattice");
	app.add_option("--beta", beta, "(inverse) coupling constant");

	// simulation options
	app.add_option("--seed", seed, "seed for the rng (default=random)");
	CLI11_PARSE(app, argc, argv);

	if (seed == -1)
		seed = std::random_device()();
	auto rng = util::xoshiro256(seed);

	auto U = randomGaugeField(geom, rng);
	auto vol = U[0].size();
	double delta = 0.1;
	std::vector<double> plaqHistory;
	for (int iter = 0; iter < 2000; ++iter)
	{
		// NOTE on conventions:
		//     * H = S(U) + 1/2 P^i P^i = S(U) - tr(P*P) = S(U) + norm2(P)
		//     * U' = P, P' = -S'(U)
		auto P = randomAlgebraField(geom, rng);

		double H_old = wilsonAction(U, beta);
		for (int mu = 0; mu < Nd; ++mu)
			H_old += norm2(P[mu]);

		GaugeField U_new;
		for (int mu = 0; mu < Nd; ++mu)
			U_new[mu] = exp(P[mu] * (0.5 * delta)) * U[mu];
		for (int mu = 0; mu < Nd; ++mu)
			P[mu]() -= (wilsonDeriv(U_new, mu, beta) * delta)();
		for (int mu = 0; mu < Nd; ++mu)
			U_new[mu] = exp(P[mu] * (0.5 * delta)) * U_new[mu];
		double H_new = wilsonAction(U_new, beta);
		for (int mu = 0; mu < Nd; ++mu)
			H_new += norm2(P[mu]);

		if (rng.uniform() < exp(H_old - H_new))
			std::swap(U, U_new);

		plaqHistory.push_back(plaquette(U));
	}

	util::Gnuplot().plotData(plaqHistory);
}
