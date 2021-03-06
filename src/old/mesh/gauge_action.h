#ifndef MESH_GAUGE_ACTION_H
#define MESH_GAUGE_ACTION_H

/**
 * Thermalization/measurement of gauge field on a fixed topology
 */

#include <random>

#include "mesh/mesh.h"

template <typename G> struct GaugeAction
{
	Mesh<G> &m;

	std::vector<std::array<LinkRef, 4>> loops4;

	std::vector<std::vector<std::array<LinkRef, 3>>> staples4;

  public:
	GaugeAction(Mesh<G> &m);

	/** one sweep of thermalization */
	void thermalize(rng_t &rng, double beta, double beta2);

	/** overrelaxation sweep */
	void overrelax();

	/** measure average loop */
	double loop4() const;
};

template <typename G>
inline GaugeAction<G>::GaugeAction(Mesh<G> &m) : m(m), loops4(m.top.loops4())
{
	staples4.reserve(m.top.nLinks());
	for (int i = 0; i < m.top.nLinks(); ++i)
		staples4.push_back(m.top.staples4(i));
}

template <typename G>
inline void GaugeAction<G>::thermalize(rng_t &rng, double beta, double beta2)
{
	// TODO: randomize order
	for (int i = 0; i < m.top.nLinks(); ++i)
	{
		G s = G::zero();
		for (auto [j, k, l] : staples4[i])
			s += m.u(j) * m.u(k) * m.u(l);
		double kappa = s.norm();

		if (kappa < 1.0e-8)
			m.u(i) = G::random(rng);
		else
			m.u(i) = s.normalize().adjoint() *
			         G::random(rng, beta * kappa, beta2 * kappa);
	}
}

template <typename G> inline void GaugeAction<G>::overrelax()
{
	// TODO: randomize order
	for (int i = 0; i < m.top.nLinks(); ++i)
	{
		G s = G::zero();
		for (auto [j, k, l] : staples4[i])
			s += m.u(j) * m.u(k) * m.u(l);
		double kappa = s.norm();

		s = s.normalize().adjoint();
		m.u(i) = s * m.u(i).adjoint() * s;
	}
}

template <typename G> inline double GaugeAction<G>::loop4() const
{
	double loop = 0;
	for (auto [i, j, k, l] : loops4)
		loop += (m.u(i) * m.u(j) * m.u(k) * m.u(l)).action();
	return loop / loops4.size();
}

#endif
