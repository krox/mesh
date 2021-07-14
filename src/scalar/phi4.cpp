#if 0

#include "scalar/phi4.h"

#include <cassert>
#include <fmt/format.h>
#include <random>
#include <vector>

#include "mesh/topology.h"
#include "util/gnuplot.h"
#include "util/hdf5.h"
#include "util/random.h"

/** one heat-bath sweep on the field */
void phi4_action::sweep()
{
	std::normal_distribution dist(0.0, 1.0);
	std::uniform_real_distribution uni_dist(0.0, 1.0);
	for (int i = 0; i < mesh.nSites(); ++i)
	{
		double c = mesh.g[i].size() + param.mass * param.mass;

		// sum of neighbors
		double rho = 0;
		for (auto j : mesh.g[i])
			rho += mesh.phi[j][0];

		while (true)
		{
			// generate with exact free distribution
			mesh.phi[i][0] = dist(rng) / sqrt(c) + rho / c;

			// accept/reject with interaction term
			double tmp = pow(mesh.phi[i][0], 4);
			if (uni_dist(rng) < exp(-param.coupling / 24.0 * tmp))
			{
				nAccept += 1;
				break;
			}
			else
				nReject += 1;
		}
	}
}

#endif
