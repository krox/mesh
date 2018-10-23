#ifndef SCALAR_PHI4_H
#define SCALAR_PHI4_H

#include "scalar/scalar.h"

struct phi4_action_param_t
{
	double mass = 0.2;
	double coupling = 0.0;
};

class phi4_action
{
  public:
	using param_t = phi4_action_param_t;

	scalar_mesh<1> &mesh;
	param_t param;

	rng_t rng;
	int64_t nAccept = 0;
	int64_t nReject = 0;

	phi4_action(scalar_mesh<1> &mesh, const param_t &param, uint64_t seed = 0)
	    : mesh(mesh), param(param), rng(seed)
	{}

	void sweep();
};

#endif
