#ifndef MESH_MESH_H
#define MESH_MESH_H

#include <array>
#include <cassert>
#include <vector>

#include "topology.h"

template <typename G> class Mesh
{
  public:
	Topology top;      // topology of the mesh
	std::vector<G> u_; // the gauge field

	/** constructor */
	explicit Mesh(Topology _top)
	    : top(std::move(_top)), u_(top.nLinks(), G::one())
	{}

	/** access to the gauge field */
	G &u(int i) { return u_[i]; }
	const G &u(int i) const { return u_[i]; }
	G u(LinkRef i) const { return i.sign ? u_[i.i].adjoint() : u_[i.i]; }
};

#endif
