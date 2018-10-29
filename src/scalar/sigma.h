#ifndef SCALAR_SIGMA_H
#define SCALAR_SIGMA_H

#include "scalar/scalar.h"

struct sigma_action_param_t
{
	double beta = 3.0;
};

/*
O(3) model:
S = β Σ_xy n_x *n_y
constaint: n_x^2 = 1
*/
class sigma_action
{
  public:
	using param_t = sigma_action_param_t;
	static constexpr size_t rep = 3;

	scalar_mesh<3> &mesh;
	param_t param;

	rng_t rng;
	int64_t nAccept = 0;
	int64_t nReject = 0;

	sigma_action(scalar_mesh<3> &mesh, const param_t &param, uint64_t seed = 0)
	    : mesh(mesh), param(param), rng(seed)
	{}

	void sweep();
};

#endif
