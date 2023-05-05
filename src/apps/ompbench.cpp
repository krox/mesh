#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "groups/su2.h"
#include "groups/sun.h"
#include "groups/u1.h"
#include "lattice/lattice.h"
#include "lattice/tensor.h"
#include "util/memory.h"
#include "util/random.h"
#include "util/stopwatch.h"
#include <algorithm>

using namespace mesh;

// prevent compiler optimizations by doing nothing but pretending to to anything
void escape(void *p) { asm volatile("" : : "g"(p) : "memory"); }
void clobber() { asm volatile("" : : : "memory"); }

// standardized output to make the log nicely human readable.
void print_line(std::string_view name, double secs, double bytes,
                double flops = 0.0 / 0.0)
{
	fmt::print(
	    "{:<25}: {:5.2f} secs, {:8.3f} GiB, {:8.3f} GiB/s, {:8.3f} GFlops/s\n",
	    name, secs, bytes / 1024. / 1024. / 1024.,
	    bytes / secs / 1024. / 1024. / 1024, flops / secs * 1e-9);
}

// create a grid of a size such that a lattice of type T will have an
// appropriate size for benchmarking
template <class T> Grid get_grid()
{
	for (int L : {8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256})
		if (size_t(L) * L * L * L * sizeof(T) >= size_t(512) * 1024 * 1024)
			return Grid({L, L, L, L});
	assert(false);
}

// benchmark `f` by running it `niter` times. returns seconds per iteration
template <class F> double benchmark(F f, int niter = 10, int nwarmup = 5)
{
	util::Stopwatch sw;
	for (int iter = -nwarmup; iter < niter; ++iter)
	{
		if (iter == 0)
			sw.start();
		f();
	}
	sw.stop();
	return sw.secs() / niter;
}

// direct memory benchmarks with very little arithmetic
void bench_memory()
{
	fmt::print("\n==================== memory ====================\n");

	using T = float;
	auto grid = get_grid<T>();
	auto a = Lattice<T>::zeros(grid);
	auto b = Lattice<T>::zeros(grid);
	double secs;

	// read

	secs = benchmark([&a]() {
		escape(a.data());
		auto s = sum(a);
		escape(&s);
	});
	print_line("sum", secs, a.bytes());

	// write

	secs = benchmark([&a]() {
		escape(a.data());
		std::memset(a.data(), 0, a.bytes());
		escape(a.data());
	});
	print_line("std::memset (1 thread)", secs, a.bytes());

	secs = benchmark([&a]() {
		escape(a.data());
		a.fill_zeros();
		escape(a.data());
	});
	print_line("fill_zeros", secs, a.bytes());

	// copy

	secs = benchmark([&a, &b]() {
		escape(a.data());
		std::memcpy(b.data(), a.data(), a.bytes());
		escape(b.data());
	});
	print_line("std::memcpy (1 thread)", secs, a.bytes() + b.bytes());

	secs = benchmark([&a, &b]() {
		escape(a.data());
		b = a;
		escape(b.data());
	});
	print_line("operator=", secs, a.bytes() + b.bytes());

	secs = benchmark([&a, &b]() {
		escape(a.data());
		lattice_apply([](auto &x, auto const &y) { x = y; }, b, a);
		escape(b.data());
	});

	print_line("lattice_apply('=')", secs, a.bytes() + b.bytes());
}

// lattice-parallel arithmetic (mostly memory bound)
void bench_axpy()
{
	fmt::print("\n==================== axpy ====================\n");

	auto run = []<class T>(std::string_view name, int complexity) {
		auto grid = get_grid<T>();

		auto a = Lattice<T>::zeros(grid);
		auto b = Lattice<T>::zeros(grid);
		auto c = Lattice<T>::zeros(grid);

		util::Stopwatch sw;
		int niter = 10;

		for (int iter = -5; iter < niter; ++iter)
		{
			if (iter == 0)
				sw.start();
			escape(b.data());
			escape(c.data());
			add_mul(a, T(1.234), c);
			escape(a.data());
		}
		sw.stop();
		double vol = a.grid().size();

		print_line(name, sw.secs() / niter, a.bytes() + b.bytes() + c.bytes(),
		           vol * complexity);
	};

	run.operator()<float>("float", 1);
	run.operator()<util::complex<float>>("complex<float>", 4);
	run.operator()<util::Matrix<util::complex<float>, 3>>("SU3<float>",
	                                                      3 * 3 * 3 * 4);
	run.operator()<util::Matrix<util::complex<float>, 4>>("SU4<float>",
	                                                      4 * 4 * 4 * 4);
	/*run.operator()<SU5<float>>("SU5<float>", 5 * 5 * 5 * 4);
	run.operator()<SU6<float>>("SU6<float>", 6 * 6 * 6 * 4);
	run.operator()<util::simd<float>>("simd<float>", util::simd<float>::size());

	run.operator()<SU3<double>>("SU3<double>", 3 * 3 * 3 * 4);
	run.operator()<SU4<double>>("SU4<double>", 4 * 4 * 4 * 4);
	run.operator()<SU5<double>>("SU5<double>", 5 * 5 * 5 * 4);
	run.operator()<SU6<double>>("SU6<double>", 6 * 6 * 6 * 4);
	run.operator()<double>("double", 1);
	run.operator()<util::complex<double>>("complex<double>", 4);
	run.operator()<util::simd<double>>("simd<double>",
	                                   util::simd<double>::size());*/
}

#if 0
template <class G> void bench_exp(std::string const &name)
{
	fmt::print("\n==================== exponential ====================\n");

	auto rng = util::xoshiro256("bench_exp");
	auto a = G::randomAlgebraElement(rng);
	G b;

	int order = 12;
	util::Stopwatch sw;
	int niter = 1'000'000;

	for (int iter = -2; iter < niter; ++iter)
	{
		if (iter == 0)
			sw.start();
		clobber();
		b = exp(a);
		clobber();
	}
	sw.stop();

	double flops = order * TensorTraits<G>::flops_mul;
	double bytes = 2.0 * sizeof(G);

	/*fmt::print("group={}, flops_mul={}, sizeof(G)={}\n", G::name(),
	           TensorTraits<G>::flops_mul, sizeof(G));*/
	fmt::print(
	    "{:<25}: {:5.2f} secs, {:8.3f} GiB, {:8.3f} GiB/s, {:8.3f} GFlops/s\n",
	    name, sw.secs(), bytes * 1e-9, bytes * 1e-9 * niter / sw.secs(),
	    flops * niter / sw.secs() * 1e-9);
}

template <class G> void bench(std::string const &name)
{
	size_t n = std::max(size_t(1), buffer_size / sizeof(G));
	auto a = util::make_aligned_unique_span<G>(n);
	auto b = util::make_aligned_unique_span<G>(n);
	auto c = util::make_aligned_unique_span<G>(n);
	escape(a.data());
	escape(b.data());
	escape(c.data());
	auto rng = util::xoshiro256(12345);
	for (size_t i = 0; i < n; ++i)
	{
		a[i] = G::randomGroupElement(rng);
		b[i] = G::randomGroupElement(rng);
	}

	util::Stopwatch sw;

	for (int iter = -2; iter < niter; ++iter)
	{
		if (iter == 0)
			sw.start();
		clobber();
#pragma omp parallel for
		for (size_t i = 0; i < n; ++i)
			c[i] = a[i] * b[i];
		clobber();
	}
	sw.stop();

	double flops = n * TensorTraits<G>::flops_mul;
	double bytes = 3.0 * n * sizeof(G);

	/*fmt::print("group={}, flops_mul={}, sizeof(G)={}\n", G::name(),
	           TensorTraits<G>::flops_mul, sizeof(G));*/
	fmt::print(
	    "{:<25}: {:5.2f} secs, {:8.3f} GiB, {:8.3f} GiB/s, {:8.3f} GFlops/s\n",
	    name, sw.secs(), bytes * 1e-9, bytes * 1e-9 * niter / sw.secs(),
	    flops * niter / sw.secs() * 1e-9);
}

template <template <class> class G> void benchGroup(std::string const &name)
{
	// bench<G<float>>(fmt::format("{}, single, scalar", name));
	// bench<G<double>>(fmt::format("{}, double, scalar", name));
	// bench<G<util::simd<float>>>(fmt::format("{}, single, simd", name));
	bench<G<util::simd<double>>>(fmt::format("{}, double, simd", name));
}
#endif

int main()
{
	util::Stopwatch sw;
	sw.start();

	bench_memory();
	bench_axpy();
	/*benchGroup<U1>("U1");
	benchGroup<SU2>("SU2");
	benchGroup<SU3>("SU3");
	benchGroup<SU4>("SU4");
	benchGroup<SU5>("SU5");
	benchGroup<SU6>("SU6");
	benchGroup<SU7>("SU7");
	benchGroup<SU8>("SU8");
	benchGroup<SU9>("SU9");
	benchGroup<SU10>("SU10");*/

	/*bench_exp<SU3<float>>("SU3, single");
	bench_exp<SU3<double>>("SU3, double");
	bench_exp<SU3<util::simd<float>>>("SU3, single, simd");
	bench_exp<SU3<util::simd<double>>>("SU3, double, simd");*/

	sw.stop();
	fmt::print("total wall time = {:.2} seconds\n", sw.secs());
}
