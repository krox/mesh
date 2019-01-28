#include "gauge/wilson.h"

#include "groups/su2.h"
#include "groups/u1.h"
#include "groups/z2.h"

#include "fmt/format.h"

template <typename G>
WilsonAction<G>::WilsonAction(GaugeMesh<G> &mesh,
                              const WilsonActionParams &param, uint64_t seed)
    : mesh(mesh), param(param), rng(seed)
{}

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
