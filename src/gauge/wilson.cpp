#include "gauge/wilson.h"

#include "groups/su2.h"
#include "groups/su3.h"
#include "groups/u1.h"
#include "groups/z2.h"

template <typename G>
WilsonAction<G>::WilsonAction(GaugeMesh<G> &mesh,
                              const WilsonActionParams &param, uint64_t seed)
    : mesh(mesh), param(param), rng(seed)
{}

// SU3 needs subgroup-algorithm
template <> void WilsonAction<SU3>::sweep()
{
	for (int i = 0; i < mesh.nLinks(); ++i)
	{
		// This is costly -> only do it once
		SU3 s;
		if (param.c1 == 0)
			s = mesh.stapleSum(i) * param.c0;
		else
			s = mesh.stapleSum(i) * param.c0 + mesh.lstapleSum(i) * param.c1;

		{
			SU2 sub = (mesh.u[i] * s).sub1();
			double alpha = param.beta * 2 / 3 * sub.norm();
			mesh.u[i] = mesh.u[i].leftMul1(SU2::random(rng, alpha) *
			                               sub.normalize().adjoint());
		}
		{
			SU2 sub = (mesh.u[i] * s).sub2();
			double alpha = param.beta * 2 / 3 * sub.norm();
			mesh.u[i] = mesh.u[i].leftMul2(SU2::random(rng, alpha) *
			                               sub.normalize().adjoint());
		}
		{
			SU2 sub = (mesh.u[i] * s).sub3();
			double alpha = param.beta * 2 / 3 * sub.norm();
			mesh.u[i] = mesh.u[i].leftMul3(SU2::random(rng, alpha) *
			                               sub.normalize().adjoint());
		}

		// this only correct numerical rounding errors -> fast version is enough
		mesh.u[i] = mesh.u[i].normalizeFast();
	}
}

// Z2, U1, SU2 can be done by direct heat-bath
template <typename G> void WilsonAction<G>::sweep()
{
	for (int i = 0; i < mesh.nLinks(); ++i)
	{
		auto s = mesh.stapleSum(i);
		double alpha = param.beta * s.norm();
		s = s.normalize();
		mesh.u[i] = (G::random(rng, alpha) * s.adjoint()).normalize();
	}
}

template class WilsonAction<Z2>;
template class WilsonAction<U1>;
template class WilsonAction<SU2>;
template class WilsonAction<SU3>;
