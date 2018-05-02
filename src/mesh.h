#ifndef MESH_MESH_H
#define MESH_MESH_H

#include <array>
#include <cassert>
#include <vector>

#include "random.h"
#include "topology.h"

template <typename G> class Mesh
{
  public:
	Topology top;      // topology of the mesh
	std::vector<G> u_; // the gauge field

	xoroshiro128plus rng;

	/** constructor */
	explicit Mesh(Topology _top)
	    : top(std::move(_top)), u_(top.nLinks(), G::one()),
	      rng(std::random_device()())
	{}

	/** access to the gauge field */
	G &u(int i) { return u_[i]; }
	const G &u(int i) const { return u_[i]; }
	G u(LinkRef i) const { return i.sign ? u_[i.i].adjoint() : u_[i.i]; }

	/** sum all staples adjacent to a given link */
	G stapleSum4(int i) const;

	/** one sweep of thermalization */
	void thermalize(double beta);

	/** measure average loop */
	double loop4() const;
};

template <typename G> inline G Mesh<G>::stapleSum4(int i) const
{
	G g = G::zero();
	for (auto [j, k, l] : top.staples4(i))
		g += u(j) * u(k) * u(l);
	return g;
}

template <typename G> inline void Mesh<G>::thermalize(double beta)
{
	// TODO: randomize order
	for (int i = 0; i < top.nLinks(); ++i)
	{
		G s = stapleSum4(i);
		double kappa = s.norm();

		// TODO: special case for kappa near zero

		u(i) = s.normalize().adjoint() * G::random(rng, beta * kappa);
	}
}

template <typename G> inline double Mesh<G>::loop4() const
{
	double loop = 0;
	int count = 0;
	for (auto [i, j, k, l] : top.loops4())
	{
		loop += (u(i) * u(j) * u(k) * u(l)).action();
		count += 1;
	}
	return loop / count;
}

#endif
