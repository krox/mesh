#include "gauge/wilson.h"

#include "groups/su2.h"
#include "groups/u1.h"
#include "groups/z2.h"

#include "fmt/format.h"

template <typename G>
wilson_action<G>::wilson_action(gauge_mesh<G> &mesh, const param_t &param,
                                uint64_t seed)
    : mesh(mesh), param(param), rng(seed)
{
	assert(mesh.geom.size() == 4);
	int nx = mesh.geom[0];
	int ny = mesh.geom[1];
	int nz = mesh.geom[2];
	int nt = mesh.geom[3];

	plaqs.reserve(6 * mesh.nSites());
	staples.resize(mesh.nLinks());
	auto staplesPos =
	    std::vector<std::vector<std::array<int, 3>>>(mesh.nLinks());
	auto staplesNeg =
	    std::vector<std::vector<std::array<int, 3>>>(mesh.nLinks());

	// (x,y,z,t,mu) -> linkId
	auto f = [&](std::array<int, 4> x, int mu) {
		x[0] = (x[0] + nx) % nx;
		x[1] = (x[1] + ny) % ny;
		x[2] = (x[2] + nz) % nz;
		x[3] = (x[3] + nt) % nt;
		return (((x[0] * ny + x[1]) * nz + x[2]) * nt + x[3]) * 4 + mu;
	};

	auto addPlaq = [&](const std::array<int, 4> &x, int mu, int nu) {
		auto xmu = x;
		xmu[mu] += 1;
		auto xnu = x;
		xnu[nu] += 1;
		int i = f(x, mu);
		int j = f(xmu, nu);
		int k = f(xnu, mu);
		int l = f(x, nu);

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
	assert((int)plaqs.size() == 6 * mesh.nSites());

	for (int i = 0; i < mesh.nLinks(); ++i)
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
}

template <typename G> G wilson_action<G>::stapleSum(int i) const
{
	auto &s = staples[i];
	auto &u = mesh.u;
	G sum = G::zero();
	sum += u[s[0][0]] * u[s[0][1]].adjoint() * u[s[0][2]].adjoint();
	sum += u[s[1][0]] * u[s[1][1]].adjoint() * u[s[1][2]].adjoint();
	sum += u[s[2][0]] * u[s[2][1]].adjoint() * u[s[2][2]].adjoint();
	sum += u[s[3][0]].adjoint() * u[s[3][1]].adjoint() * u[s[3][2]];
	sum += u[s[4][0]].adjoint() * u[s[4][1]].adjoint() * u[s[4][2]];
	sum += u[s[5][0]].adjoint() * u[s[5][1]].adjoint() * u[s[5][2]];
	return sum;
}

template <typename G> void wilson_action<G>::sweep()
{
	for (int i = 0; i < mesh.nLinks(); ++i)
	{
		auto s = stapleSum(i);
		double alpha = param.beta * s.norm();
		// fmt::print("{}\n", alpha);
		s = s.normalize();
		mesh.u[i] = (G::random(rng, alpha) * s.adjoint()).normalize();
	}
}

template <typename G> double wilson_action<G>::action() const
{
	auto &u = mesh.u;
	double sum = 0.0;
	for (auto [i, j, k, l] : plaqs)
	{
		// fmt::print("{} {} {} {}\n", i, j, k, l);
		sum += (u[i] * u[j] * u[k].adjoint() * u[l].adjoint()).action();
	}
	return sum / plaqs.size();
}

template class wilson_action<Z2>;
template class wilson_action<U1>;
template class wilson_action<SU2>;