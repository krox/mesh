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

} // namespace mesh
