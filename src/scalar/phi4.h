#pragma once

#if 0

#include "scalar/scalar.h"

struct phi4_action_param_t
{
	double mass = 0.2;
	double coupling = 0.0;

	// stupid hack
	double beta = 0.0 / 0.0;
	double mu = 0.0 / 0.0;
};

/*
Scalar ϕ^4 theory:
S = 1/2 Σ_xy (ϕ_x-ϕ_y)^2 +  m^2/2 Σ_x ϕ_x^2 + g/24 Σ_x ϕ_x^4
*/
class phi4_action
{
  public:
	using param_t = phi4_action_param_t;
	static constexpr size_t rep = 1;

	ScalarMesh<1> &mesh;
	param_t param;

	rng_t rng;
	int64_t nAccept = 0;
	int64_t nReject = 0;

	phi4_action(ScalarMesh<1> &mesh, const param_t &param, uint64_t seed = 0)
	    : mesh(mesh), param(param), rng(seed)
	{}

	void sweep();
	void cluster() { assert(false); }

	double action() const { return 0.0 / 0.0; }
	double magnetization() const { return 0.0 / 0.0; }
	double phaseAngle() const { return 0.0; }
};

#endif
