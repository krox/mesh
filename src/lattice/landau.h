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

	// derivative of term. should go to zero by gauge fixing
	double condition() const
	{
		int Nd = U.grid().ndim();
		double vol = U.grid().size();

		auto d = Lattice<vG>::zeros(U.grid());
		for (int mu = 0; mu < Nd; ++mu)
		{
			d += U[mu] * adj(cshift(g, mu, 1));
			d += adj(cshift(g * U[mu], mu, -1));
		}
		return norm2(projectOnAlgebra(g * d)) * (1.0 / (Nd * vol));
	}

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

	// saddle point of quadratic given by three points
	static double solve_quadratic(double x1, double x2, double x3, double y1,
	                              double y2, double y3)
	{
		return 0.5 *
		       ((x2 * x2 - x3 * x3) * y1 - (x1 * x1 - x3 * x3) * y2 +
		        (x1 * x1 - x2 * x2) * y3) /
		       ((x2 - x3) * y1 - (x1 - x3) * y2 + (x1 - x2) * y3);
	}

	double line_search(Lattice<vG> const &F, double eps) const
	{
		double base = (*this)();

		double y1 = (*this)(exp(F * eps) * g);
		while (y1 <= base)
		{
			eps *= 0.5;
			y1 = (*this)(exp(F * eps) * g);
			if (eps < 1e-10)
				return 0.0;
		}

		double y2 = (*this)(exp(F * (eps * 1.2)) * g);

		if (y2 > y1)
			return 1.2 * eps;

		double y3 = (*this)(exp(F * (eps * 0.8)) * g);

		return solve_quadratic(eps, 1.2 * eps, 0.8 * eps, y1, y2, y3);
	}

	void run(int max_iter)
	{
		double eps = 3000.0;

		// non-linear CG
		Lattice<vG> last(U.grid());
		double lastNorm = 0.0;
		for (int iter = 0; iter < max_iter; ++iter)
		{
			auto cond = condition();
			if (verbose)
				fmt::print("iter = {}, cond = {}, term = {}, eps = {},\n",
				           iter + 1, cond, (*this)(), eps);
			if (cond < 1e-10)
				return;

			auto F = deriv();
			double norm = real(sumTrace(F * F));
			// NOTE: only start the CG after a few iterations of SD

			// Fletcher-Reeves
			// double beta = norm / lastNorm;

			// Polak-Ribière with automatic restart
			double beta = (norm - real(sumTrace(F * last))) / lastNorm;

			last = F;
			if (iter > 20 && beta > 0)
				F += last * beta;

			eps = line_search(F, eps);
			g = exp(F * eps) * g;
			reunitize(g);

			lastNorm = norm;
		}
	}
};

} // namespace mesh
