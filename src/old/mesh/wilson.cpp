
#include "mesh/wilson.h"

#include "groups/su2.h"
#include "groups/u1.h"
#include "groups/z2.h"
#include "util/lattice.h"

namespace {

/** Element-wise multiplication of a and b. Result stored in buf */
template <typename G>
Lattice<G, 4> multiply(std::vector<G> &buf, Lattice<const G, 4> a,
                       Lattice<const G, 4> b)
{
	assert(a.shape() == b.shape());
	assert(buf.empty());
	buf.resize(a.size());
	auto r = Lattice<G, 4>(buf, a.shape());
	eval(r, [](const G &a, const G &b) { return a * b; }, a, b);
	return r;
}

/** sum(action(a * b * c.adj * d.adj)) */
template <typename G>
double plaqSum(Lattice<const G, 4> a, Lattice<const G, 4> b,
               Lattice<const G, 4> c, Lattice<const G, 4> d)
{
	assert(a.shape() == b.shape() && b.shape() == c.shape() &&
	       c.shape() == d.shape());
	return contract(
	    [](const G &a, const G &b, const G &c, const G &d) {
		    return ((a * b) * (d * c).adjoint()).action();
	    },
	    a, b, c, d);
}

} // namespace

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
		bb[mu].resize(u[mu].shape()[mu] + 1); // index 0 is unused
		bb[mu][1] = u[mu];
		for (int n = 2; n <= u[mu].shape()[mu]; ++n)
		{
			bufs.emplace_back();
			bb[mu][n] =
			    multiply(bufs.back(), u[mu], bb[mu][n - 1].shift(mu, +1));
		}
	}

	/** compute loops itself */
	std::vector<std::tuple<int, int, double>> r;
	for (int n = 1; n <= maxN; ++n)
		for (int t = 1; t <= m.top.geom[3]; ++t)
		{
			double s = 0;
			for (int mu = 0; mu < 3; ++mu)
				s += plaqSum(bb[mu][n], bb[3][t].shift(mu, n),
				             bb[mu][n].shift(3, t), bb[3][t]);
			s /= u[0].size() * 3;
			r.push_back({n, t, s});
		}

	return r;
}

using namespace std;
template vector<tuple<int, int, double>> wilson<Z2>(const Mesh<Z2> &, int);
template vector<tuple<int, int, double>> wilson<U1>(const Mesh<U1> &, int);
template vector<tuple<int, int, double>> wilson<SU2>(const Mesh<SU2> &, int);
