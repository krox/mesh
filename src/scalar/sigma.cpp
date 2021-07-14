#if 0

#include "scalar/sigma.h"

#include <cassert>
#include <fmt/format.h>
#include <random>
#include <vector>

#include "mesh/topology.h"
#include "util/gnuplot.h"
#include "util/hdf5.h"
#include "util/random.h"

/** one heat-bath sweep on the field */
void sigma_action::sweep()
{
	std::normal_distribution dist(0.0, 1.0);
	std::uniform_real_distribution uni_dist(0.0, 1.0);
	Scalar one;
	one[0] = 1.0;
	one[1] = 0.0;
	one[2] = 0.0;

	for (int i = 0; i < mesh.nSites(); ++i)
	{
		// "staple-sum"
		Scalar rho;
		rho[0] = rho[1] = rho[2] = 0.0;
		for (auto j : mesh.g[i])
			rho += mesh.phi[j];
		double alpha = util::length(rho) * param.beta;

		while (true)
		{
			if (alpha < 1.0e-10)
			{
				mesh.phi[i] = util::uniform_sphere_distribution<Scalar>()(rng);
			}
			else
			{
				mesh.phi[i] =
				    util::exponential_sphere_distribution<Scalar>(alpha)(rng);
				Scalar refl = (one - util::normalize(rho));
				if (util::length(refl) >= 1.0e-8)
				{
					refl = util::normalize(refl);
					mesh.phi[i] =
					    mesh.phi[i] - refl * util::dot(refl, mesh.phi[i]) * 2.0;
				}
			}

			// real part of chemical potential term
			if (uni_dist(rng) < exp(-0.5 * param.beta * param.mu * param.mu *
			                        mesh.phi[i][0] * mesh.phi[i][0]))
			{
				nAccept += 1;
				break;
			}
			else
				nReject += 1;
		}
	}
}

void sigma_action::sweepMesh(int nSwaps)
{
	for (int iter = 0; iter < nSwaps; ++iter)
	{
	}
}

double sigma_action::action() const
{
	// NOTE: this assumes mu = 0
	double sum = 0.0;
	for (int i = 0; i < mesh.nSites(); ++i)
		for (auto j : mesh.g[i])
			sum += util::dot(mesh.phi[i], mesh.phi[j]);
	return 4.0 - sum / mesh.nSites();
}

#endif
