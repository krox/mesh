#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "lattice/gauge.h"
#include "lattice/hmc.h"
#include "lattice/landau.h"

using namespace mesh;

template <typename vG> void testImpl(Coordinate geom, util::xoshiro256 &rng)
{
	fmt::print("\n========== testing G={}, Nd={}, simd={} ==========\n",
	           vG::name(), geom.size(), TensorTraits<vG>::simd_width);
	auto const &g = Grid::make(geom, TensorTraits<vG>::simd_width);

	auto f = Lattice<vG>(g);
	auto F = GaugeField<vG>(g);
	auto U = GaugeField<vG>(g);
	randomAlgebraField(f, rng);
	randomAlgebraField(F, rng);
	randomGaugeField(U, rng);
	fmt::print("normalizaiton of norm2(random algebra): {}\n",
	           norm2(F) * (1.0 / (g.size() * g.ndim() * vG::dim())) / 0.5);
	fmt::print("normalizaiton of trace(random algebra^2): {}\n",
	           sumTrace(F * F) * (1.0 / (g.size() * g.ndim() * vG::dim())) /
	               (-0.5));

	{
		auto landau = Landau(U);
		auto old = landau();

		double eps = 0.0001;
		double expected = -2.0 * eps * real(sumTrace(f * landau.deriv()));
		landau.g = exp(f * eps) * landau.g;
		fmt::print("derivative of Landau gauge condition: {}\n",
		           (landau() - old) / expected);
	}
}

int main(int argc, char **argv)
{
	CLI::App app{
	    "Test some operations of lattice gauge theory, mostly to verify that "
	    "normalizations and sign-conventions and such are consistent."};
	CLI11_PARSE(app, argc, argv);

	auto rng = util::xoshiro256(std::random_device()());
	for (auto &geom : {Coordinate{8, 8, 8, 8}, Coordinate{8, 8, 8}})
	{
		testImpl<U1<util::simd<double>>>(geom, rng);
		testImpl<SU2<util::simd<double>>>(geom, rng);
		testImpl<SU3<util::simd<double>>>(geom, rng);
		testImpl<U1<util::simd<float>>>(geom, rng);
		testImpl<SU2<util::simd<float>>>(geom, rng);
		testImpl<SU3<util::simd<float>>>(geom, rng);
	}
}