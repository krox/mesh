#include "scalar/ising.h"

#include <cassert>
#include <fmt/format.h>
#include <random>
#include <vector>

#include "mesh/topology.h"
#include "util/random.h"

// NOTE: the field may be incorrectly zero-initialized (instead of +1/-1)

void IsingAction::sweep(mesh_t &mesh, rng_t &rng)
{
	for (int i = 0; i < mesh.nSites(); ++i)
	{
		// sum of neighbors
		double rho = 0;
		for (auto const &link : mesh.top.graph[i])
			rho += mesh.phi[link.to][0];
		rho *= params.beta;

		// probability
		double p = exp(rho) / (exp(rho) + exp(-rho));
		mesh.phi[i][0] = std::bernoulli_distribution(p)(rng) ? 1.0 : -1.0;
		nAccept += 1;
	}
}

void IsingAction::cluster(mesh_t &mesh, rng_t &rng)
{
	auto dist = std::bernoulli_distribution(1.0 - exp(-2 * params.beta));
	std::vector<int> q;

	// start cluster a random site
	int start = std::uniform_int_distribution<int>(0, mesh.nSites() - 1)(rng);
	double old = mesh.phi[start][0];
	mesh.phi[start][0] = -old;
	q.push_back(start);

	while (!q.empty())
	{
		int i = q.back();
		q.pop_back();

		for (auto const &link : mesh.top.graph[i])
		{
			if (mesh.phi[link.to][0] != old) // wrong spin, or already flipped
				continue;

			// extend the cluster with probability p = 1 - exp(-2 beta)
			if (dist(rng))
			{
				mesh.phi[link.to][0] = -old;
				q.push_back(link.to);
			}
		}
	}
}

double IsingAction::action(mesh_t &mesh) const
{
	double sum = 0.0;
	for (int i = 0; i < mesh.nSites(); ++i)
		for (auto const &link : mesh.top.graph[i])
			sum += mesh.phi[i][0] * mesh.phi[link.to][0];
	return params.beta * 0.5 * sum / mesh.nSites();
}

double IsingAction::magnetization(mesh_t &mesh) const
{
	double sum = 0.0;
	for (int i = 0; i < mesh.nSites(); ++i)
		sum += mesh.phi[i][0];
	return sum / mesh.nSites();
}
