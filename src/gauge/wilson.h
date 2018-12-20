#ifndef GAUGE_WILSON_H
#define GAUGE_WILSON_H

#include "gauge/gauge.h"

#include "util/random.h"

struct wilson_action_param_t
{
	double beta = 0.0;
};

/*
wilson plaquette action:
S = -β/N Σ_ijkl Re Tr (U_i U_j U_k^H U_l^H)
*/
template <typename _G> class wilson_action
{
  public:
	using param_t = wilson_action_param_t;
	using G = _G;

	gauge_mesh<G> &mesh;
	param_t param;

	rng_t rng;

	// plaqs: u(i)*u(j)*u(k).adj*u(l).adj
	// staples 0,1,2: u(j) * u(k).adj * u(l).adj
	//         3,4,5: u(j).adj * u(k).adj * u(l)
	std::vector<std::array<int, 4>> plaqs;
	std::vector<std::array<std::array<int, 3>, 6>> staples;

	wilson_action(gauge_mesh<G> &mesh, const param_t &param, uint64_t seed = 0);

	/** sum of staples of link i */
	G stapleSum(int i) const;

	/** heat-bath sweep */
	void sweep();

	/** is this possible to implement? */
	void cluster() { assert(false); }

	/** average (normalized) plaquette */
	double action() const;
};

#endif
