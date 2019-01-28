#ifndef GAUGE_WILSON_H
#define GAUGE_WILSON_H

#include <cassert>

#include "gauge/gauge.h"

#include "util/random.h"

/** parameters of Wilson action */
struct WilsonActionParams
{
	double beta = 0.0;
	// future: adjoint action, improvement coefficients
};

/*
Wilson plaquette action:
S = -β/N Σ_ijkl Re Tr (U_i U_j U_k^H U_l^H)
*/
template <typename _G> class WilsonAction
{
  public:
	using G = _G;

	GaugeMesh<G> &mesh;
	WilsonActionParams param;

	rng_t rng;

	WilsonAction(GaugeMesh<G> &mesh, const WilsonActionParams &param,
	             uint64_t seed);

	/** heat-bath sweep */
	void sweep();

	/** is this possible to implement? */
	void cluster() { assert(false); }
};

#endif
