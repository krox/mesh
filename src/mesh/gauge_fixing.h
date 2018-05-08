#ifndef MESH_GAUGE_FIXING_H
#define MESH_GAUGE_FIXING_H

/**
 * Landau gauge fixing.
 */

/** relax a single link */
template <typename G>
void gaugeRelax(const Mesh<G> &m, std::vector<G> &g, int a)
{
	G v = G::zero();
	for (auto [b, i] : m.top.graph[a])
		v += m.u(i) * g[b].adjoint();
	if (v.norm() < 1.0e-8)
		return;
	g[a] = v.normalize().adjoint();
}

/** one sweep of gauge relaxation */
template <typename G> void gaugeRelax(const Mesh<G> &m, std::vector<G> &g)
{
	assert((int)g.size() == m.top.nSites());
	for (int i = 0; i < m.top.nSites(); ++i)
		gaugeRelax(m, g, i);
}

/** average link */
template <typename G>
inline double avgLink(const Mesh<G> &m, const std::vector<G> &g)
{
	assert(m.top.nSites() == (int)g.size());
	double sum = 0;
	for (int i = 0; i < m.top.nLinks(); ++i)
	{
		int a = m.top.links[i].from;
		int b = m.top.links[i].to;
		sum += (g[a] * m.u(i) * g[b].adjoint()).action();
	}
	return sum / m.top.nLinks();
}

/** measure gauge condition */
template <typename G>
inline double gaugeCondition(const Mesh<G> &m, const std::vector<G> &g)
{
	double cond = 0;
	for (int a = 0; a < m.top.nSites(); ++a)
	{
		G diff = G::zero();
		for (auto [b, i] : m.top.graph[a])
			diff += (g[a] * m.u(i) * g[b].adjoint()).algebra();
		cond += diff.norm();
	}
	return cond / m.top.nSites();
}

/** inplace rotate gauge field */
template <typename G>
inline void gaugeRotate(Mesh<G> &m, const std::vector<G> &g)
{
	assert(m.top.nSites() == (int)g.size());
	for (int i = 0; i < m.top.nLinks(); ++i)
	{
		int a = m.top.links[i].from;
		int b = m.top.links[i].to;
		m.u(i) = g[a] * m.u(i) * g[b].adjoint();
	}
}

template <typename G>
inline std::vector<G> gaugeFix(const Mesh<G> &m, double eps, int maxIter)
{
	// start rotation field at unity
	std::vector<G> g(m.top.nSites(), G::one());

	// measure the non-fixed field
	double gaugeCond = gaugeCondition(m, g);

#ifdef LOG_GAUGEFIXING
	double gaugeTerm = avgLink(m, g);
	std::cout << "iter = 0, gaugeCond = " << gaugeCond
	          << ", link = " << gaugeTerm << std::endl;
#endif

	for (int iter = 5; iter <= maxIter; iter += 5)
	{
		for (int i = 0; i < 5; ++i)
			gaugeRelax(m, g);

		gaugeCond = gaugeCondition(m, g);

#ifdef LOG_GAUGEFIXING
		gaugeTerm = avgLink(m, g);
		std::cout << "iter = " << iter << ", gaugeCond = " << gaugeCond
		          << ", link = " << gaugeTerm << std::endl;
#endif

		if (gaugeCond <= eps)
			return g;
	}

	throw "gauge fixing did not converge";
}

#endif
