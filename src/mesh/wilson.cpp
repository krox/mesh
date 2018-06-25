
#include "mesh/wilson.h"

#include "mesh/su2.h"
#include "mesh/u1.h"
#include "mesh/z2.h"
#include "util/lattice.h"

template <typename G>
std::vector<std::tuple<int, int, double>> wilson(const Mesh<G> &m, int maxN)
{
	assert(m.top.top == "periodic4");
	auto u = m.asLattice4();

	/** compute building blocks (straight lines) */
	std::vector<std::vector<G>> bufs;
	std::vector<Lattice<const G, 4>> bb[4];
	for (int mu = 0; mu < 4; ++mu)
	{
		bb[mu].resize(maxN + 1); // index 0 is unused
		bb[mu][1] = u[mu];
		for (int n = 2; n <= maxN; ++n)
		{
			bufs.emplace_back();
			bufs.back().resize(u[mu].size());
			bb[mu][n] = (u[mu] * bb[mu][n - 1].shift(mu, +1)).eval(bufs.back());
		}
	}

	/** compute loops itself */
	std::vector<std::tuple<int, int, double>> r;
	for (int na = 1; na <= maxN; ++na)
		for (int nb = na; nb <= maxN; ++nb)
		{
			double s = 0;
			for (int mu = 0; mu < 4; ++mu)
				for (int nu = 0; nu < 4; ++nu)
				{
					if (mu == nu)
						continue;
					if (na == nb && mu > nu)
						continue;

					auto plaq = bb[mu][na] * bb[nu][nb].shift(mu, na) *
					            bb[mu][na].shift(nu, nb).map(&G::adjoint) *
					            bb[nu][nb].map(&G::adjoint);
					s += plaq.map(&G::action).sum();
				}
			s /= u[0].size() * (na == nb ? 6 : 12);
			r.push_back({na, nb, s});
		}
	return r;
}

using namespace std;
template vector<tuple<int, int, double>> wilson<Z2>(const Mesh<Z2> &, int);
template vector<tuple<int, int, double>> wilson<U1>(const Mesh<U1> &, int);
template vector<tuple<int, int, double>> wilson<SU2>(const Mesh<SU2> &, int);
