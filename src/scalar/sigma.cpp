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
				mesh.phi[i] = mesh.phi[i] - refl * refl.dot(mesh.phi[i]) * 2.0;
			}
		}
	}
}
