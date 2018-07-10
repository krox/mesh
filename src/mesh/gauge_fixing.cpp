#include "mesh/gauge_fixing.h"

#include <fmt/format.h>

#include "mesh/su2.h"
#include "mesh/u1.h"

template <typename G> double GaugeFixing<G>::gaugeTerm() const
{
	double sum = 0;
	for (int i = 0; i < m.top.nLinks(); ++i)
	{
		int a = m.top.links[i].from;
		int b = m.top.links[i].to;
		sum += (g[a] * m.u(i) * g[b].adjoint()).action();
	}
	return sum / m.top.nLinks();
}

template <typename G> double GaugeFixing<G>::gaugeCondition() const
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

template <typename G> void GaugeFixing<G>::relaxSite(int a)
{
	G v = G::zero();
	for (auto [b, i] : m.top.graph[a])
		v += m.u(i) * g[b].adjoint();
	if (v.norm() < 1.0e-8)
		return;
	g[a] = v.normalize().adjoint();
}

template <typename G> void GaugeFixing<G>::relaxSweep()
{
	for (int i = 0; i < m.top.nSites(); ++i)
		relaxSite(i);
}

template <typename G> void GaugeFixing<G>::relax(double eps, int maxIter)
{
	// measure the non-fixed field
	double cond = gaugeCondition();
	double term = 0;

	if (verbose)
	{
		term = gaugeTerm();
		fmt::print("i = {}, cond = {}, link = {}\n", 0, cond, term);
	}

	for (int iter = 5; iter <= maxIter; iter += 5)
	{
		for (int i = 0; i < 5; ++i)
			relaxSweep();

		cond = gaugeCondition();

		if (verbose)
		{
			term = gaugeTerm();
			fmt::print("i = {}, cond = {}, link = {}\n", iter, cond, term);
		}

		if (cond <= eps)
			return;
	}

	throw std::runtime_error("gauge fixing did not converge");
}

template class GaugeFixing<U1>;
template class GaugeFixing<SU2>;
