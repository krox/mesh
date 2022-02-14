#pragma once

#include "lattice/gauge.h"

namespace mesh {

template <typename vG> struct Landau
{
	GaugeField<vG> const &U;
	Lattice<vG> g;
	bool verbose = false;

	Landau(GaugeField<vG> const &U_) : U(U_), g(Lattice<vG>::ones(U_.grid())) {}

	// maximized by gauge fixing

	double operator()(Lattice<vG> const &h) const
	{
		int Nd = U.grid().ndim();
		double vol = U.grid().size();

		double s = 0;
		for (int mu = 0; mu < Nd; ++mu)
			s += real(sumTrace(h * U[mu] * adj(cshift(h, mu, 1))));
		return s / (vG::Nc() * Nd * vol);
	}

	double operator()() const { return (*this)(g); }

	// (algebra-valued) derivative of term
	Lattice<vG> deriv() const
	{
		int Nd = U.grid().ndim();
		double vol = U.grid().size();

		auto d = Lattice<vG>::zeros(U.grid());
		for (int mu = 0; mu < Nd; ++mu)
		{
			d += U[mu] * adj(cshift(g, mu, 1));
			d += adj(cshift(g * U[mu], mu, -1));
		}
		return projectOnAlgebra(g * d) * (-0.5 / (vG::Nc() * Nd * vol));
	}

	void run(int max_iter)
	{
		double eps = 1.0;

		for (int iter = 0; iter < max_iter; ++iter)
		{
			auto cond = (*this)();
			fmt::print("iter = {}, cond = {}, eps = {}\n", iter + 1, (*this)(),
			           eps);
			auto F = deriv();

			double condNew = (*this)(exp(F * (1.0 * eps)) * g);
			if (condNew < cond)
			{
				fmt::print("WARNING: can no longer increase!!");
				return;
			}

			double tmp = (*this)(exp(F * (2.0 * eps)) * g);

			auto solve = [](double x1, double x2, double x3, double y1,
			                double y2, double y3) {
				return 0.5 *
				       ((x2 * x2 - x3 * x3) * y1 - (x1 * x1 - x3 * x3) * y2 +
				        (x1 * x1 - x2 * x2) * y3) /
				       ((x2 - x3) * y1 - (x1 - x3) * y2 + (x1 - x2) * y3);
			};

			if (tmp > condNew)
			{
				eps *= 2;
			}
			else
			{
				eps = solve(0, eps, 2.0 * eps, cond, condNew, tmp);
			}
			g = exp(F * eps) * g;
			reunitize(g);
		}
	}
	/*
	    // derivative of term. should go to zero by gauge fixing
	    double condition() const
	    {
	        double c = 0;
	        int Nd = U.grid().ndim();
	        double vol = U.grid().size();

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
	*/
};

} // namespace mesh
