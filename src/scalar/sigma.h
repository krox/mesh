#ifndef SCALAR_SIGMA_H
#define SCALAR_SIGMA_H

#include "scalar/scalar.h"

struct sigma_action_param_t
{
	double beta = 3.0;
	double mu = 0.0; // not really implemented.
};

/*
O(3) model:
S = -β ( Σ_xy n_x * n_y - μ^2/2 Σ_x n3^2 - iμ Σ_x n1(x)*(n2(x+t)-n2(x-t)) )
constaint: n_x^2 = 1
*/
class sigma_action
{
  public:
	using param_t = sigma_action_param_t;
	static constexpr size_t rep = 3;

	ScalarMesh<3> &mesh;
	using Scalar = ScalarMesh<3>::Scalar;
	param_t param;

	rng_t rng;
	int64_t nAccept = 0;
	int64_t nReject = 0;

	sigma_action(ScalarMesh<3> &mesh, const param_t &param, uint64_t seed = 0)
	    : mesh(mesh), param(param), rng(seed)
	{}

	/** heat-bath sweep (phase quenched) */
	void sweep();

	void cluster() { assert(false); }

	/** real part of (negative) action */
	double action() const;

	double magnetization() const { return 0.0 / 0.0; }

	/** imaginary part of (negative) action */
	double phaseAngle() const;
};

#endif
