#ifndef SCALAR_ISING_H
#define SCALAR_ISING_H

#include "scalar/scalar.h"

struct ising_action_param_t
{
	double beta = 0.0 / 0.0; // nearest-neighbour coupling
	double mu = 0.0 / 0.0;   // not implemented yet
};

/*
Ising model:
S = -β Σ_xy ϕ_x ϕ_y
*/
class ising_action
{
  public:
	using param_t = ising_action_param_t;
	static constexpr size_t rep = 1;

	ScalarMesh<1> &mesh;
	param_t param;

	rng_t rng;
	int64_t nAccept = 0;
	int64_t nReject = 0;

	ising_action(ScalarMesh<1> &mesh, const param_t &param, uint64_t seed = 0)
	    : mesh(mesh), param(param), rng(seed)
	{}

	void sweep();   // heat-bath sweep
	void cluster(); // one cluster update

	double action() const;
	double magnetization() const;
};

#endif
