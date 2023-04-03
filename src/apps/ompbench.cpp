#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "groups/su2.h"
#include "groups/sun.h"
#include "groups/u1.h"
#include "lattice/tensor.h"
#include "util/memory.h"
#include "util/random.h"
#include "util/stopwatch.h"
#include <algorithm>

using namespace mesh;

template <class T> using SU7 = SUN<T, 7>;
template <class T> using SU8 = SUN<T, 8>;
template <class T> using SU9 = SUN<T, 9>;
template <class T> using SU10 = SUN<T, 10>;

// prevent compiler optimizations by doing nothing but pretending to to anything
void escape(void *p) { asm volatile("" : : "g"(p) : "memory"); }
void clobber() { asm volatile("" : : : "memory"); }

// size of objects (in bytes) to benchmark.
static constexpr size_t buffer_size = 1024 * 1024 * 128;
static constexpr int niter = 20;

// tiny buffer, many iterations -> everything in cache should hit compute limit
// static constexpr size_t buffer_size = 1;
// static constexpr int niter = 2'000'000;

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

int main()
{
	util::Stopwatch sw;
	sw.start();

	benchGroup<U1>("U1");
	benchGroup<SU2>("SU2");
	benchGroup<SU3>("SU3");
	benchGroup<SU4>("SU4");
	benchGroup<SU5>("SU5");
	benchGroup<SU6>("SU6");
	benchGroup<SU7>("SU7");
	benchGroup<SU8>("SU8");
	benchGroup<SU9>("SU9");
	benchGroup<SU10>("SU10");

	sw.stop();
	fmt::print("total wall time = {:.2} seconds\n", sw.secs());
}
