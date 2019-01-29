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

	// add one plaquette
	auto addPlaq = [&](const std::array<int, 4> &x, int mu, int nu) {
		int i = lid(x, mu);
		int j = lid(x, mu, nu);
		int k = lid(x, nu, mu);
		int l = lid(x, nu);

		// ijKL = lkJI
		plaqs.push_back({i, j, k, l});
		staplesPos[i].push_back({j, k, l});
		staplesNeg[j].push_back({k, l, i});
		staplesPos[l].push_back({k, j, i});
		staplesNeg[k].push_back({j, i, l});
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

	rects.reserve(24 * nSites());
	lstaples.resize(nLinks());
	auto lstaples1 = std::vector<std::vector<std::array<int, 5>>>(nLinks());
	auto lstaples2 = std::vector<std::vector<std::array<int, 5>>>(nLinks());
	auto lstaples3 = std::vector<std::vector<std::array<int, 5>>>(nLinks());

	// add two rects
	auto addRects = [&](const std::array<int, 4> &x, int mu, int nu) {
		{
			// mu mu nu -mu -mu -nu
			int a = lid(x, mu);
			int b = lid(x, mu, mu);
			int c = lid(x, mu, mu, nu);
			int d = lid(x, mu, nu, mu);
			int e = lid(x, nu, mu);
			int f = lid(x, nu);

			// abcDEF = fedCBA
			rects.push_back({a, b, c, d, e, f});
			lstaples1[a].push_back({b, c, d, e, f});
			lstaples2[b].push_back({c, d, e, f, a});
			lstaples3[c].push_back({d, e, f, a, b});
			lstaples1[f].push_back({e, d, c, b, a});
			lstaples2[e].push_back({d, c, b, a, f});
			lstaples3[d].push_back({c, b, a, f, e});
		}
		{
			// mu nu nu -mu -nu -nu
			int a = lid(x, mu);
			int b = lid(x, mu, nu);
			int c = lid(x, mu, nu, nu);
			int d = lid(x, nu, nu, mu);
			int e = lid(x, nu, nu);
			int f = lid(x, nu);

			// abcDEF = fedCBA
			rects.push_back({a, b, c, d, e, f});
			lstaples1[a].push_back({b, c, d, e, f});
			lstaples2[b].push_back({c, d, e, f, a});
			lstaples3[c].push_back({d, e, f, a, b});
			lstaples1[f].push_back({e, d, c, b, a});
			lstaples2[e].push_back({d, c, b, a, f});
			lstaples3[d].push_back({c, b, a, f, e});
		}
	};

	for (int x = 0; x < nx; ++x)
		for (int y = 0; y < ny; ++y)
			for (int z = 0; z < nz; ++z)
				for (int t = 0; t < nt; ++t)
					for (int mu = 0; mu < 4; ++mu)
						for (int nu = 0; nu < 4; ++nu)
							if (mu != nu)
								addRects({x, y, z, t}, mu, nu);
	assert((int)rects.size() == 24 * nSites());

	for (int i = 0; i < nLinks(); ++i)
	{
		assert(lstaples1[i].size() == 12);
		assert(lstaples2[i].size() == 12);
		assert(lstaples3[i].size() == 12);
		for (int k = 0; k < 12; ++k)
		{
			lstaples[i][k] = lstaples1[i][k];
			lstaples[i][k + 12] = lstaples2[i][k];
			lstaples[i][k + 24] = lstaples3[i][k];
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
	for (int k = 0; k < 12; ++k)
		sum += u[s[k][0]] * u[s[k][1]] * u[s[k][2]].adjoint() *
		       u[s[k][2]].adjoint() * u[s[k][2]].adjoint();
	for (int k = 12; k < 24; ++k)
		sum += u[s[k][0]] * u[s[k][1]].adjoint() * u[s[k][2]].adjoint() *
		       u[s[k][2]].adjoint() * u[s[k][2]];
	for (int k = 24; k < 36; ++k)
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
		s = (s - s.adjoint()).traceless() * (0.5 * alpha);

		s = G::one() + s + s * s * 0.5 + s * s * s * (1.0 / 6) +
		    s * s * s * s * (1.0 / 24) +
		    s * s * s * s * s * (1.0 / 120); // TODO: actual exponential

		r.u[i] = (s * u[i]).normalize();
	}
	return r;
}

template <typename G> double GaugeMesh<G>::topCharge() const
{
	/** NOTE: plaqs are ordered [x,y,z,t,{xy,xz,xt,yz,yt,zt}] */

	// sum up clover ( = sum of 4 plaqs) at every site/orientation
	auto clover = std::vector<std::array<G, 6>>(nSites());
	for (auto &a : clover)
		a.fill(G::zero());
	for (int pi = 0; pi < nPlaqs(); ++pi)
	{
		auto [i, j, k, l] = top->plaqs[pi]; // link ids of this plaq
		int o = pi % 6;                     // orientation of this plaq

		// this plaquette is part of for different clovers
		clover[i / 4][o] += u[i] * u[j] * u[k].adjoint() * u[l].adjoint();
		clover[j / 4][o] += u[j] * u[k].adjoint() * u[l].adjoint() * u[i];
		clover[k / 4][o] += u[k].adjoint() * u[l].adjoint() * u[i] * u[j];
		clover[l / 4][o] += u[l].adjoint() * u[i] * u[j] * u[k].adjoint();
	}

	// sum up topological charge
	double Q = 0.0;
	for (int i = 0; i < nSites(); ++i)
	{
		auto &c = clover[i];
		Q += (c[0].antisym() * c[5].antisym()).action(); // F(xy) F(zt)
		Q -= (c[1].antisym() * c[4].antisym()).action(); // F(xz) F(yt)
		Q += (c[2].antisym() * c[3].antisym()).action(); // F(xt) F(yz)
	}

	return Q; // TODO: some constant factor is missing
}

template class GaugeMesh<Z2>;
template class GaugeMesh<U1>;
template class GaugeMesh<SU2>;
template class GaugeMesh<SU3>;
