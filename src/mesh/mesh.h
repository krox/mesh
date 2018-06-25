#ifndef MESH_MESH_H
#define MESH_MESH_H

#include <array>
#include <cassert>
#include <vector>

#include "mesh/topology.h"
#include "util/lattice.h"
#include "util/random.h"
#include "util/span.h"

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

	span<double> rawLinks();
	span<const double> rawLinks() const;
	span<const double> rawLinksConst() const;

	std::array<Lattice<const G, 4>, 4> asLattice4() const;
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

template <typename G> inline span<double> Mesh<G>::rawLinks()
{
	static_assert(G::repSize() * sizeof(double) == sizeof(G));
	return span((double *)u_.data(), G::repSize() * u_.size());
}

template <typename G> inline span<const double> Mesh<G>::rawLinks() const
{
	static_assert(G::repSize() * sizeof(double) == sizeof(G));
	return span((const double *)u_.data(), G::repSize() * u_.size());
}

template <typename G> inline span<const double> Mesh<G>::rawLinksConst() const
{
	static_assert(G::repSize() * sizeof(double) == sizeof(G));
	return span((const double *)u_.data(), G::repSize() * u_.size());
}

template <typename G>
inline std::array<Lattice<const G, 4>, 4> Mesh<G>::asLattice4() const
{
	auto g = top.geom;
	assert(g.size() == 4);
	auto all = Lattice<const G, 5>(u_, {g[0], g[1], g[2], g[3], 4});
	std::array<Lattice<const G, 4>, 4> u;
	for (int i = 0; i < 4; ++i)
		u[i] = all.slice(4, i);
	return u;
}

#endif
