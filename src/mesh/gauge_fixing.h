#ifndef MESH_GAUGE_FIXING_H
#define MESH_GAUGE_FIXING_H

/**
 * Landau gauge fixing.
 */

#include <vector>

#include "mesh/mesh.h"

template <typename G> class GaugeFixing
{
  public:
	const Mesh<G> &m; // config to be fixed
	std::vector<G> g; // rotation which (approximatively) fixes m

	bool verbose = false;

	/** constructor */
	GaugeFixing(const Mesh<G> &m) : m(m), g(m.top.nSites(), G::one()) {}

	/** measurement */
	double gaugeTerm() const;      // average link (maximal in Landau gauge)
	double gaugeCondition() const; // gauge condition (=0 in Landau gauge)

	/** simple relaxation algorithm */
	void relaxSite(int a);               // relax one site
	void relaxSweep();                   // relax all sites
	void relax(double eps, int maxIter); // relax until cond() <= eps
};

#endif
