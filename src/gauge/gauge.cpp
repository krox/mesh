#include "gauge/gauge.h"

#include "gauge/wilson.h"
#include "groups/su2.h"
#include "groups/su3.h"
#include "groups/u1.h"
#include "groups/z2.h"
#include "xtensor/xview.hpp"
#include <fmt/format.h>

GaugeTopology::GaugeTopology(const std::vector<int> &geom)
{
	assert(geom.size() == 4);
	int nx = geom[0];
	int ny = geom[1];
	int nz = geom[2];
	int nt = geom[3];
	this->geom = {nx, ny, nz, nt};

	plaqs.reserve(6 * nSites());
	staples.resize(nLinks());
	auto staplesPos = std::vector<std::vector<std::array<int, 3>>>(nLinks());
	auto staplesNeg = std::vector<std::vector<std::array<int, 3>>>(nLinks());
	clovers.resize(nSites());

	// add one plaquette
	auto addPlaq = [&](const std::array<int, 4> &x, int mu, int nu) {
		int a = sid(x);
		int b = sid(x, mu);
		int c = sid(x, mu, nu);
		int d = sid(x, nu);

		int i = a * 4 + mu;
		int j = b * 4 + nu;
		int k = d * 4 + mu;
		int l = a * 4 + nu;

		// ijKL = lkJI
		plaqs.push_back({i, j, k, l});
		staplesPos[i].push_back({j, k, l});
		staplesNeg[j].push_back({k, l, i});
		staplesPos[l].push_back({k, j, i});
		staplesNeg[k].push_back({j, i, l});

		int o = (nu - 1) + (mu == 0 ? 0 : mu == 1 ? 2 : 3);
		clovers[a][o][0] = {i, j, k, l};
		clovers[b][o][1] = {j, k, l, i};
		clovers[c][o][2] = {k, j, i, j};
		clovers[d][o][3] = {l, i, j, k};
	};

	for (int x = 0; x < nx; ++x)
		for (int y = 0; y < ny; ++y)
			for (int z = 0; z < nz; ++z)
				for (int t = 0; t < nt; ++t)
					for (int mu = 0; mu < 4; ++mu)
						for (int nu = mu + 1; nu < 4; ++nu)
							addPlaq({x, y, z, t}, mu, nu);
	assert((int)plaqs.size() == 6 * nSites());

	for (int i = 0; i < nLinks(); ++i)
	{
		assert(staplesPos[i].size() == 3);
		assert(staplesNeg[i].size() == 3);
		staples[i][0] = staplesPos[i][0];
		staples[i][1] = staplesPos[i][1];
		staples[i][2] = staplesPos[i][2];
		staples[i][3] = staplesNeg[i][0];
		staples[i][4] = staplesNeg[i][1];
		staples[i][5] = staplesNeg[i][2];
	}

	rects.reserve(12 * nSites());
	lstaples.resize(nLinks());
	auto lstaples1 = std::vector<std::vector<std::array<int, 5>>>(nLinks());
	auto lstaples2 = std::vector<std::vector<std::array<int, 5>>>(nLinks());
	auto lstaples3 = std::vector<std::vector<std::array<int, 5>>>(nLinks());

	// add one rects (mu mu nu -mu -mu -nu)
	auto addRect = [&](const std::array<int, 4> &x, int mu, int nu) {
		int a = sid(x) * 4 + mu;
		int b = sid(x, mu) * 4 + mu;
		int c = sid(x, mu, mu) * 4 + nu;
		int d = sid(x, mu, nu) * 4 + mu;
		int e = sid(x, nu) * 4 + mu;
		int f = sid(x) * 4 + nu;

		// abcDEF = fedCBA
		rects.push_back({a, b, c, d, e, f});
		lstaples1[a].push_back({b, c, d, e, f});
		lstaples2[b].push_back({c, d, e, f, a});
		lstaples3[c].push_back({d, e, f, a, b});
		lstaples1[f].push_back({e, d, c, b, a});
		lstaples2[e].push_back({d, c, b, a, f});
		lstaples3[d].push_back({c, b, a, f, e});
	};

	for (int x = 0; x < nx; ++x)
		for (int y = 0; y < ny; ++y)
			for (int z = 0; z < nz; ++z)
				for (int t = 0; t < nt; ++t)
					for (int mu = 0; mu < 4; ++mu)
						for (int nu = 0; nu < 4; ++nu)
							if (mu != nu)
								addRect({x, y, z, t}, mu, nu);
	assert((int)rects.size() == 12 * nSites());

	for (int i = 0; i < nLinks(); ++i)
	{
		assert(lstaples1[i].size() == 6);
		assert(lstaples2[i].size() == 6);
		assert(lstaples3[i].size() == 6);
		for (int k = 0; k < 6; ++k)
		{
			lstaples[i][k] = lstaples1[i][k];
			lstaples[i][k + 6] = lstaples2[i][k];
			lstaples[i][k + 12] = lstaples3[i][k];
		}
	}
}

template <typename G> G GaugeMesh<G>::stapleSum(int i) const
{
	auto &s = top->staples[i];
	G sum = G::zero();
	sum += u[s[0][0]] * u[s[0][1]].adjoint() * u[s[0][2]].adjoint();
	sum += u[s[1][0]] * u[s[1][1]].adjoint() * u[s[1][2]].adjoint();
	sum += u[s[2][0]] * u[s[2][1]].adjoint() * u[s[2][2]].adjoint();
	sum += u[s[3][0]].adjoint() * u[s[3][1]].adjoint() * u[s[3][2]];
	sum += u[s[4][0]].adjoint() * u[s[4][1]].adjoint() * u[s[4][2]];
	sum += u[s[5][0]].adjoint() * u[s[5][1]].adjoint() * u[s[5][2]];
	return sum;
}

template <typename G> G GaugeMesh<G>::lstapleSum(int i) const
{
	auto &s = top->lstaples[i];
	G sum = G::zero();
	for (int k = 0; k < 6; ++k)
		sum += u[s[k][0]] * u[s[k][1]] * u[s[k][2]].adjoint() *
		       u[s[k][2]].adjoint() * u[s[k][2]].adjoint();
	for (int k = 6; k < 12; ++k)
		sum += u[s[k][0]] * u[s[k][1]].adjoint() * u[s[k][2]].adjoint() *
		       u[s[k][2]].adjoint() * u[s[k][2]];
	for (int k = 12; k < 18; ++k)
		sum += u[s[k][0]].adjoint() * u[s[k][1]].adjoint() *
		       u[s[k][2]].adjoint() * u[s[k][2]] * u[s[k][2]];
	return sum;
}

template <typename G> double GaugeMesh<G>::plaqSum() const
{
	double sum = 0;
	for (auto [i, j, k, l] : top->plaqs)
		sum += (u[i] * u[j] * u[k].adjoint() * u[l].adjoint()).action();
	return sum;
}

template <typename G> double GaugeMesh<G>::rectSum() const
{
	double sum = 0;
	for (auto [a, b, c, d, e, f] : top->rects)
		sum += (u[a] * u[b] * u[c] * u[d].adjoint() * u[e].adjoint() *
		        u[f].adjoint())
		           .action();
	return sum;
}

template <typename G> GaugeMesh<G> GaugeMesh<G>::smearCool() const
{
	GaugeMesh<G> r(top);
	for (int i = 0; i < nLinks(); ++i)
		r.u[i] = stapleSum(i).adjoint().normalize();
	return r;
}

template <typename G> GaugeMesh<G> GaugeMesh<G>::smearAPE(double a) const
{
	GaugeMesh<G> r(top);
	for (int i = 0; i < nLinks(); ++i)
		r.u[i] = (u[i] * (1.0 - a) + stapleAvg(i).adjoint() * a).normalize();
	return r;
}

template <typename G> GaugeMesh<G> GaugeMesh<G>::smearEXP(double alpha) const
{
	GaugeMesh<G> r(top);
	for (int i = 0; i < nLinks(); ++i)
	{
		G s = u[i] * stapleSum(i);
		s = (s - s.adjoint()).traceless() * (-0.5 * alpha);

		s = G::one() + s + (s * s) * 0.5 + (s * s) * s * (1.0 / 6) +
		    (s * s) * (s * s) * (1.0 / 24) +
		    (s * s) * (s * s) * s * (1.0 / 120); // TODO: actual exponential

		r.u[i] = (s * u[i]).normalize();
	}
	return r;
}

template <typename G> G GaugeMesh<G>::clover(int i, int o) const
{
	assert(0 <= o && o < 6);
	G s = G::zero();

	auto &c = top->clovers[i][o];
	s += u[c[0][0]] * u[c[0][1]] * u[c[0][2]].adjoint() * u[c[0][3]].adjoint();
	s += u[c[1][0]] * u[c[1][1]].adjoint() * u[c[1][2]].adjoint() * u[c[1][3]];
	s += u[c[2][0]].adjoint() * u[c[2][1]].adjoint() * u[c[2][2]] * u[c[2][3]];
	s += u[c[3][0]].adjoint() * u[c[3][1]] * u[c[3][2]] * u[c[3][3]].adjoint();
	return s;
}

template <typename G> double GaugeMesh<G>::topCharge() const
{
	// sum up topological charge
	double Q = 0.0;
	for (int i = 0; i < nSites(); ++i)
	{
		{
			G g = clover(i, 0);
			G h = clover(i, 5);
			Q += ((g - g.adjoint()).traceless() * (h - h.adjoint()).traceless())
			         .action();
		}
		{
			G g = clover(i, 1);
			G h = clover(i, 4);
			Q -= ((g - g.adjoint()).traceless() * (h - h.adjoint()).traceless())
			         .action();
		}
		{
			G g = clover(i, 2);
			G h = clover(i, 3);
			Q += ((g - g.adjoint()).traceless() * (h - h.adjoint()).traceless())
			         .action();
		}
	}
	double factor = 1.0 / (32.0 * 3.14159 * 3.14169); // definition of Q_top
	factor *= 24.0 / 3.0;  // the epsilon summation has 24 terms, we use only 3
	factor *= 0.25;        // "hermitian part" is missing a factor 1/2
	factor *= 0.25 * 0.25; // clover needs normalization
	factor *= 9.0;         // we used ".action" when we meant ".trace"
	// NOTE: no normalization by volume
	return Q * factor;
}

template class GaugeMesh<Z2>;
template class GaugeMesh<U1>;
template class GaugeMesh<SU2>;
template class GaugeMesh<SU3>;
