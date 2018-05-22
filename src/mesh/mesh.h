#ifndef MESH_MESH_H
#define MESH_MESH_H

#include <array>
#include <cassert>
#include <vector>

#include "mesh/topology.h"
#include "util/random.h"

template <typename G> class Mesh
{
  public:
	Topology top;      // topology of the mesh
	std::vector<G> u_; // the gauge field

	/** constructor */
	explicit Mesh(Topology _top)
	    : top(std::move(_top)), u_(top.nLinks(), G::one())
	{}

	void initOrdered();
	void initRandom(rng_t &rng);
	void initMixed(rng_t &rng);

	/** access to the gauge field */
	G &u(int i) { return u_[i]; }
	const G &u(int i) const { return u_[i]; }
	G u(LinkRef i) const { return i.sign ? u_[i.i].adjoint() : u_[i.i]; }
};

template <typename G> inline void Mesh<G>::initOrdered()
{
	for (G &g : u_)
		g = G::one();
}

template <typename G> inline void Mesh<G>::initRandom(rng_t &rng)
{
	for (G &g : u_)
		g = G::random(rng);
}

template <typename G> inline void Mesh<G>::initMixed(rng_t &rng)
{
	for (size_t i = 0; i < u_.size() / 2; ++i)
		u(i) = G::one();
	for (size_t i = u_.size() / 2; i < u_.size(); ++i)
		u(i) = G::random(rng);
}

#endif
