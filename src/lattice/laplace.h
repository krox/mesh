#pragma once

/**
 * Lapalce operator for scalar lattice.
 */

#include "lattice/lattice.h"

namespace mesh {

// Discrete Laplace operator. This is symmetric-negative-definite for all h.
class LaplaceOperator
{
	double h_; // (isotropic) lattice spacing

  public:
	LaplaceOperator() : h_(1.0) {}
	explicit LaplaceOperator(double h) : h_(h) { assert(h > 0); }

	template <typename vT> void apply(Lattice<vT> &v, Lattice<vT> const &u)
	{
		assert(compatible(v, u));
		assert(v.data() != u.data());
		int Nd = v.grid().ndim();

		// TODO: this is very suboptimal (producing too many temporaries.
		//       should theoretically produce none at all)
		v = u * (-2.0 * Nd);
		for (int mu = 0; mu < Nd; ++mu)
		{
			v += cshift(u, mu, 1);
			v += cshift(u, mu, -1);
		}
		v *= 1.0 / h_ / h_;
	}

	template <typename vT>
	void apply_diagonal_inverse(Lattice<vT> &v, Lattice<vT> const &u)
	{
		assert(compatible(v, u));
		int Nd = v.grid().ndim();

		// TODO: this is very suboptimal (producing atemporary)
		v = u * (1.0 / (-2.0 * Nd / h_ / h_));
	}

	template <typename vT> Lattice<vT> apply(Lattice<vT> const &u)
	{
		auto v = Lattice<vT>(u.grid());
		apply(v, u);
		return v;
	}

	template <typename vT>
	Lattice<vT> apply_diagonal_inverse(Lattice<vT> const &u)
	{
		auto v = Lattice<vT>(u.grid());
		apply_diagonal_inverse(v, u);
		return v;
	}
};

} // namespace mesh
