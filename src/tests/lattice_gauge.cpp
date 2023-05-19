#include "catch2/catch_test_macros.hpp"

#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "lattice/gauge.h"
#include "lattice/hmc.h"
#include "lattice/landau.h"

using namespace mesh;

template <typename G> void testImpl(Coordinate geom, util::xoshiro256 &rng)
{
	fmt::print("\n========== testing G={}, Nd={} ==========\n",
	           GaugeTraits<G>::name(), geom.size());
	auto g = Grid(geom);

	auto f = Lattice<G>(g);
	auto F = GaugeField<G>(g);
	auto U = GaugeField<G>(g);
	random_algebra_field(f, rng);
	random_algebra_field(F, rng);
	random_gauge_field(U, rng);
	fmt::print("normalizaiton of norm2(random algebra): {}\n",
	           norm2(F) *
	               (1.0 / (g.size() * g.ndim() * GaugeTraits<G>::dim())) / 0.5);
	fmt::print("normalizaiton of trace(random algebra^2): {}\n",
	           sum_real_trace(F * F) *
	               (1.0 / (g.size() * g.ndim() * GaugeTraits<G>::dim())) /
	               (-0.5));

	{
		auto landau = Landau(U);
		auto old = landau();

		double eps = 0.0001;
		double expected = -2.0 * eps * sum_real_trace(f * landau.deriv());
		landau.g = exp(f * eps) * landau.g;
		fmt::print("derivative of Landau gauge condition: {}\n",
		           (landau() - old) / expected);
	}
}

TEST_CASE("lattice gauge", "[lattice][gauge]")
{
	auto rng = util::xoshiro256(std::random_device()());
	for (auto &geom : {Coordinate{8, 8, 8, 8}, Coordinate{8, 8, 8}})
	{
		testImpl<U1<double>>(geom, rng);
		testImpl<SU2<double>>(geom, rng);
		testImpl<SU3<double>>(geom, rng);
		testImpl<U1<float>>(geom, rng);
		testImpl<SU2<float>>(geom, rng);
		testImpl<SU3<float>>(geom, rng);
	}
}
