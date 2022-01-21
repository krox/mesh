#include "lattice/hmc.h"

namespace mesh {

std::vector<double> makeDeltas(std::string_view scheme, double epsilon,
                               int substeps)
{
	// Langevin -> HMC convention conversion
	double delta = sqrt(2 * epsilon) / substeps;

	std::vector<double> deltas_raw;
	if (scheme == "2lf")
	{
		deltas_raw = {0.5, 1.0, 0.5};
	}
	else if (scheme == "2mn")
	{
		double lamb = 0.19318332750378361;
		deltas_raw = {lamb, 0.5, 1.0 - 2.0 * lamb, 0.5, lamb};
	}
	else if (scheme == "4mn")
	{
		double rho = 0.1786178958448091;
		double theta = -0.06626458266981843;
		double lamb = 0.7123418310626056;
		deltas_raw = {
		    rho,        lamb,  theta, 0.5 - lamb, 1.0 - 2.0 * (rho + theta),
		    0.5 - lamb, theta, lamb,  rho};
	}
	else
		throw std::runtime_error(fmt::format("unknown scheme '{}'", scheme));

	std::vector<double> deltas;
	deltas.reserve((deltas_raw.size() - 1) * substeps + 1);
	for (int i = 0; i < substeps; ++i)
	{
		if (i == 0)
			deltas.insert(deltas.end(), deltas_raw.begin(), deltas_raw.end());
		else
		{
			deltas.back() *= 2;
			deltas.insert(deltas.end(), deltas_raw.begin() + 1,
			              deltas_raw.end());
		}
	}
	for (double &d : deltas)
		d *= delta;
	return deltas;
}

template <typename vG> void runHmc_impl(HmcParams const &params)
{
	auto rng = util::xoshiro256(params.seed == -1 ? std::random_device()()
	                                              : params.seed);

	auto const &g =
	    Grid::make(Coordinate(params.geom.begin(), params.geom.end()), 4);
	int Nd = g.ndim();

	auto U = makeGaugeField<vG>(g);
	randomGaugeField(U, rng);
	auto P = makeGaugeField<vG>(g);

	auto U_new = U;
	auto vol = g.size();
	auto deltas = makeDeltas(params.scheme, params.epsilon, params.substeps);

	// tracking some observables
	int64_t nAccept = 0, nReject = 0;
	std::vector<double> plaqHistory;
	std::vector<double> deltaHHistory;

	std::optional<util::Gnuplot> plot;
	if (params.doPlot)
		plot.emplace();

	for (size_t iter : util::ProgressRange(params.count))
	{
		// NOTE on conventions:
		//     * H = S(U) + 1/2 P^i P^i = S(U) - tr(P*P) = S(U) + norm2(P)
		//     * U' = P, P' = -S'(U)
		randomAlgebraField(P, rng);

		double H_old = wilsonAction(U, params.beta);
		for (int mu = 0; mu < Nd; ++mu)
			H_old += norm2(P[mu]);

		U_new = U;
		runHmd(U_new, P, params.beta, deltas);
		reunitize(U_new);
		double H_new = wilsonAction(U_new, params.beta);
		for (int mu = 0; mu < Nd; ++mu)
			H_new += norm2(P[mu]);
		auto deltaH = H_new - H_old;

		// metropolis step
		if (rng.uniform() < exp(-deltaH))
		{
			++nAccept;
			std::swap(U, U_new);
		}
		else
			++nReject;

		plaqHistory.push_back(plaquette(U));
		deltaHHistory.push_back(deltaH);

		if ((iter + 1) % (100 / params.substeps) == 0 && plot)
		{
			plot->clear();
			plot->plotData(
			    util::span(plaqHistory)
			        .slice(plaqHistory.size() / 10, plaqHistory.size()));
		}
	}

	fmt::print("plaquette = {} +- {}\n", util::mean(plaqHistory),
	           sqrt(util::variance(plaqHistory) / plaqHistory.size()));
	fmt::print("acceptance = {:.2f}\n", nAccept / double(nAccept + nReject));
	fmt::print("<exp(-dH)> = {:.4f}\n",
	           util::mean(deltaHHistory, [](double x) { return exp(-x); }));
	fmt::print("lattice allocs: {}\n", latticeAllocCount);
	fmt::print("time in cshift: {:.2f}\n", swCshift.secs());

	if (plot)
	{
		plot->clear();
		plot->plotData(plaqHistory);
	}
}

// explicit instantiations
template void runHmc_impl<U1<util::simd<float>>>(HmcParams const &);
template void runHmc_impl<SU2<util::simd<float>>>(HmcParams const &);
template void runHmc_impl<SU3<util::simd<float>>>(HmcParams const &);
template void runHmc_impl<U1<util::simd<double>>>(HmcParams const &);
template void runHmc_impl<SU2<util::simd<double>>>(HmcParams const &);
template void runHmc_impl<SU3<util::simd<double>>>(HmcParams const &);

} // namespace mesh
