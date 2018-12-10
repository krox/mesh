#include "scalar/ising.h"

#include <cassert>
#include <fmt/format.h>
#include <random>
#include <vector>

#include "mesh/topology.h"
#include "util/random.h"

/** one heat-bath sweep on the field */
void ising_action::sweep()
{
	for (int i = 0; i < mesh.nSites(); ++i)
	{
		// sum of neighbors
		double rho = 0;
		for (auto j : mesh.g[i])
			rho += mesh.phi[j][0];
		rho *= param.beta;

		// probability
		double p = exp(rho) / (exp(rho) + exp(-rho));
		mesh.phi[i][0] = std::bernoulli_distribution(p)(rng) ? 1.0 : -1.0;
		nAccept += 1;
	}
}

double ising_action::action() const
{
	double sum = 0.0;
	for (int i = 0; i < mesh.nSites(); ++i)
		for (auto j : mesh.g[i])
			sum += mesh.phi[i][0] * mesh.phi[j][0];
	return param.beta * 0.5 * sum / mesh.nSites();
}

double ising_action::magnetization() const
{
	double sum = 0.0;
	for (int i = 0; i < mesh.nSites(); ++i)
		sum += mesh.phi[i][0];
	return sum / mesh.nSites();
}
