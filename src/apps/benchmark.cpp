#include "CLI/CLI.hpp"
#include "Eigen/Dense"
#include "fmt/format.h"
#include "lattice/gauge.h"
#include "lattice/hmc.h"
#include "util/gnuplot.h"
#include "util/hash.h"

using namespace mesh;
using Eigen::MatrixXd, Eigen::ArrayXd;

// prevent compiler optimizations by doing nothing but pretending to to anything
void escape(void *p) { asm volatile("" : : "g"(p) : "memory"); }
void clobber() { asm volatile("" : : : "memory"); }

std::pair<double, double> testEigenMatmul(size_t target_mem)
{
	size_t n = (size_t)std::sqrt(target_mem / 3. / sizeof(double));
	auto a = MatrixXd(n, n);
	auto b = MatrixXd(n, n);
	auto c = MatrixXd(n, n);
	a.setRandom();
	b.setRandom();

	double best = std::numeric_limits<double>::infinity();
	for (int i = 0; i < 5; ++i)
	{
		util::Stopwatch sw;
		sw.start();
		escape(&a(0));
		escape(&b(0));
		c = a * b;
		escape(&c(0));
		sw.stop();
		best = std::min(best, sw.secs());
	}

	double mem = 3. * n * n * sizeof(double);
	double ops = n * n * n;

	return {mem, ops / best};
}

std::pair<double, double> testEigenVecmul(size_t target_mem)
{
	size_t n = (size_t)(target_mem / 3. / sizeof(double));
	auto a = ArrayXd(n);
	auto b = ArrayXd(n);
	auto c = ArrayXd(n);
	a.setRandom();
	b.setRandom();

	double best = std::numeric_limits<double>::infinity();
	for (int i = 0; i < 5; ++i)
	{
		util::Stopwatch sw;
		sw.start();
		escape(&a(0));
		escape(&b(0));
		c = a * b;
		escape(&c(0));
		sw.stop();
		best = std::min(best, sw.secs());
	}

	double mem = 3. * n * sizeof(double);
	double ops = n;

	return {mem, ops / best};
}

std::pair<double, double> testSU2Mul(size_t target_mem)
{
	int n = (int)pow(target_mem / 3. / sizeof(SU2<double>), 0.25);
	if (n % 2)
		n += 1;
	auto &grid = Grid::make_double({n, n, n, n});
	using vT = SU2<util::simd<double>>;

	auto a = Lattice<vT>(grid);
	auto b = Lattice<vT>(grid);
	auto c = Lattice<vT>(grid);
	util::xoshiro256 rng = {};
	randomGaugeField(a, rng);
	randomGaugeField(b, rng);

	double best = std::numeric_limits<double>::infinity();
	for (int i = 0; i < 5; ++i)
	{
		util::Stopwatch sw;
		sw.start();
		escape(a.data());
		escape(b.data());
		lattice_apply([](auto &cc, auto &aa, auto &bb) { cc = aa * bb; }, c, a,
		              b);
		escape(c.data());
		sw.stop();
		best = std::min(best, sw.secs());
	}

	double mem = 3 * grid.size() * sizeof(SU2<double>);
	double ops = grid.size() * 16.;

	return {mem, ops / best};
}

int main()
{
	fmt::print("using {} threads in Eigen\n", Eigen::nbThreads());
	std::vector<double> xs_matmul, ys_matmul;
	std::vector<double> xs_vecmul, ys_vecmul;
	std::vector<double> xs_su2, ys_su2;

	auto plot = util::Gnuplot();
	plot.setLogScaleX();

	for (size_t target = 1'000; target < 1'000'000'000; target += target / 10)
	{
		{
			auto [mem, flops] = testEigenMatmul(target);
			xs_matmul.push_back(mem / 1024. / 1024.);
			ys_matmul.push_back(flops / 1.e9);
		}
		{
			auto [mem, flops] = testEigenVecmul(target);
			xs_vecmul.push_back(mem / 1024. / 1024.);
			ys_vecmul.push_back(flops / 1.e9);
		}
		{
			auto [mem, flops] = testSU2Mul(target);
			xs_su2.push_back(mem / 1024. / 1024.);
			ys_su2.push_back(flops / 1.e9);
		}

		if (xs_matmul.size() >= 2)
		{
			plot.clear();
			plot.plotData(xs_matmul, ys_matmul, "Eigen matmul");
			plot.plotData(xs_vecmul, ys_vecmul, "Eigen vecmul");
			plot.plotData(xs_su2, ys_su2, "SU(2)");
		}
	}
}
