#include "gauge/hmc.h"

#include <vector>

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

// NOTE: we hide the implementation behind a compilation boundary, mostly in
//       order to improve comilation times during development

template <typename G> Hmc<G>::Hmc(Grid const &g) : g(g), U(g), P(g) {}

// reset the gauge field to a random config
template <typename G> void Hmc<G>::randomizeGaugeField()
{
	random_gauge_field(U, rng);
}

// new gaussian momenta
template <typename G> void Hmc<G>::randomizeMomenta()
{
	// NOTE on conventions:
	//     * H = S(U) + 1/2 P^i P^i = S(U) - tr(P*P) = S(U) + norm2(P)
	//     * U' = P, P' = -S'(U)
	random_algebra_field(P, rng);
}

// generate momenta -> run a trajectory -> accept/reject it -> measure
// (NOTE: even if rejected, old momenta are destroyed)
template <typename G>
void Hmc<G>::runHmcUpdate(double beta, std::vector<double> const &deltas)
{
	util::Stopwatch sw;
	sw.start();

	// generate new momenta
	randomizeMomenta();

	// make proposal
	double H_old = wilsonAction(U, beta) + norm2(P);
	U_new = U;
	runHmd(U_new, P, beta, deltas);
	reunitize(U_new); // no idea if this is the best place to put it
	double H_new = wilsonAction(U_new, beta) + norm2(P);
	auto deltaH = H_new - H_old;

	// metropolis step
	if (rng.uniform() < exp(-deltaH))
	{
		accept_history.push_back(1.0);
		std::swap(U, U_new);
	}
	else
		accept_history.push_back(0.0);

	sw.stop();

	// track some observables
	plaq_history.push_back(plaquette(U));
	deltaH_history.push_back(deltaH);
	time_history.push_back(sw.secs());
}

template class Hmc<U1<float>>;
template class Hmc<U1<double>>;
template class Hmc<SU2<float>>;
template class Hmc<SU2<double>>;
template class Hmc<SU3<float>>;
template class Hmc<SU3<double>>;

} // namespace mesh
