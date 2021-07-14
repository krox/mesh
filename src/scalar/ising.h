#pragma once

#include "scalar/scalar.h"

struct IsingActionParams
{
	double beta = 0.0 / 0.0; // nearest-neighbour coupling
	                         // double mu = 0.0 / 0.0;   // chemical potential
};

/*
Ising model:
S = -β Σ_xy ϕ_x ϕ_y
with ϕ = ±1
*/
class IsingAction
{
  public:
	static constexpr size_t rep = 1;
	using params_t = IsingActionParams;
	using mesh_t = ScalarMesh<1>;

	params_t params;

	int64_t nAccept = 0;
	int64_t nReject = 0;

	IsingAction(params_t const &params) : params(params) {}

	// direct updates: heat-bath and cluster algorithm
	void sweep(mesh_t &mesh, rng_t &rng);
	void cluster(mesh_t &mesh, rng_t &rng);

	double action(mesh_t &mesh) const;
	double magnetization(mesh_t &mesh) const;
};
