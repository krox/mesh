#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "lattice/gauge.h"
#include "lattice/hmc.h"

using namespace mesh;

template <typename vG> void testImpl(Coordinate geom)
{
	fmt::print("\n========== testing G={}, Nd={}, simd={} ==========\n",
	           vG::name(), geom.size(), TensorTraits<vG>::simd_width);
	auto rng = util::xoshiro256(0);
	auto const &g = Grid::make(geom, TensorTraits<vG>::simd_width);

	auto F = GaugeField<vG>(g);
	randomAlgebraField(F, rng);
	fmt::print("normalizaiton of norm2(random algebra): {}\n",
	           norm2(F) * (1.0 / (g.size() * g.ndim() * vG::dim())) / 0.5);
	fmt::print("normalizaiton of trace(random algebra^2): {}\n",
	           sumTrace(F * F) * (1.0 / (g.size() * g.ndim() * vG::dim())) /
	               (-0.5));
}

int main(int argc, char **argv)
{
	CLI::App app{
	    "Test some operations of lattice gauge theory, mostly to verify that "
	    "normalizations and sign-conventions and such are consistent."};
	CLI11_PARSE(app, argc, argv);

	for (auto &geom : {Coordinate{8, 8, 8, 8}, Coordinate{8, 8, 8}})
	{
		testImpl<U1<util::simd<double>>>(geom);
		testImpl<SU2<util::simd<double>>>(geom);
		testImpl<SU3<util::simd<double>>>(geom);
		testImpl<U1<util::simd<float>>>(geom);
		testImpl<SU2<util::simd<float>>>(geom);
		testImpl<SU3<util::simd<float>>>(geom);
	}
}
