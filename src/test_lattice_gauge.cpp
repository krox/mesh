#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "lattice/gauge.h"
#include "lattice/hmc.h"

using namespace mesh;

template <typename vG> void testImpl(Coordinate geom)
{
	fmt::print("\n========== testing G={}, Nd={}, simd={} ==========\n",
	           vG::name(), geom.size(), TensorTraits<vG>::simdWidth);
	auto rng = util::xoshiro256(0);
	auto const &g = Grid::make(geom, TensorTraits<vG>::simdWidth);

	auto F = makeGaugeField<vG>(g);
	randomAlgebraField(F, rng);
	fmt::print("normalizaiton of norm2(random algebra): {}\n",
	           norm2(F) / (double)(g.size() * g.ndim() * vG::dim()) / 0.5);
	fmt::print("normalizaiton of trace(random algebra^2): {}\n",
	           sumTrace(F * F) / (double)(g.size() * g.ndim() * vG::dim()) /
	               (-0.5));
}

int main(int argc, char **argv)
{
	CLI::App app{
	    "Test some operations of lattice gauge theory, mostly to verify that "
	    "normalizations and sign-conventions and such are consistent."};
	CLI11_PARSE(app, argc, argv);

	testImpl<U1<util::simd<double>>>({8, 8, 8, 8});
	testImpl<SU2<util::simd<double>>>({8, 8, 8, 8});
	testImpl<SU3<util::simd<double>>>({8, 8, 8, 8});
}
