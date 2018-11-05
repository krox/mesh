#include "scalar/sigma.h"

#include <cassert>
#include <fmt/format.h>
#include <random>
#include <vector>

#include "mesh/topology.h"
#include "util/fft.h"
#include "util/gnuplot.h"
#include "util/io.h"
#include "util/random.h"

/** one heat-bath sweep on the field */
void sigma_action::sweep()
{
	std::normal_distribution dist(0.0, 1.0);
	std::uniform_real_distribution uni_dist(0.0, 1.0);
	for (int i = 0; i < mesh.nSites(); ++i)
	{
		// "staple-sum"
		Scalar<3> rho = Scalar<3>::zero();
		for (auto j : mesh.g[i])
			rho += mesh.phi[j];
		double alpha = rho.norm() * param.beta;

		while (true)
		{
			if (alpha < 1.0e-10)
			{
				mesh.phi[i] = Scalar<3>::randomSphere(rng);
			}
			else
			{
				mesh.phi[i] = Scalar<3>::randomSphere(rng, alpha);
				Scalar<3> refl = (Scalar<3>::one() - rho.normalize());
				if (refl.norm() >= 1.0e-8)
				{
					refl = refl.normalize();
					mesh.phi[i] =
					    mesh.phi[i] - refl * refl.dot(mesh.phi[i]) * 2.0;
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

double sigma_action::action() const
{
	// NOTE: this assumes mu = 0
	double sum = 0.0;
	for (int i = 0; i < mesh.nSites(); ++i)
		for (auto j : mesh.g[i])
			sum += mesh.phi[i].dot(mesh.phi[j]);
	return 4.0 - sum / mesh.nSites();
}

double sigma_action::phaseAngle() const
{
	double sum = 0.0;
	for (int i = 0; i < mesh.nSites(); ++i)
	{
		sum += mesh.phi[i][1] * mesh.phi[mesh.timeStep[i]][2];
		sum -= mesh.phi[i][2] * mesh.phi[mesh.timeStep[i]][1];
	}
	return -param.mu * sum;
}
