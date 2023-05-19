#pragma once

#include "fmt/format.h"
#include "lattice/grid.h"
#include "lattice/tensor.h"
#include "util/hdf5.h"
#include "util/ndarray.h"
#include "util/span.h"
#include "util/stopwatch.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <omp.h>
#include <type_traits>

namespace mesh {

inline int64_t latticeAllocCount = 0;
inline util::Stopwatch swCshift;

void parallel_memcpy(void *, void const *, size_t);
void parallel_memset(void *, int, size_t);

template <typename T> class Lattice
{
  public:
	// sanity checks on the types
	static_assert(std::is_same_v<T, std::decay_t<T>>);
	static_assert(std::is_trivially_copyable_v<T>);
	// static_assert(sizeof(Object) == Object::size() * sizeof(Real));
	// static_assert(std::is_same_v<Real, float> || std::is_same_v<Real,
	// double>);

	using value_type = T;

  private:
	Grid grid_ = {}; // this is dimension 1, size 0
	util::unique_span<T> data_ = {};

  public:
	Lattice() = default;

	// allocates memory but does not initialize anything
	explicit Lattice(Grid const &g) : grid_(g)
	{
		++latticeAllocCount;
		data_ = util::make_aligned_unique_span<T>(grid().size());
	}

	// allocates memory and set it to zero
	//   (zero bits, does not take the object type into account)
	static Lattice zeros(Grid const &g)
	{
		auto a = Lattice(g);
		a.fill_zeros();
		return a;
	}

	void fill_zeros() { parallel_memset(data(), 0, bytes()); }

	Grid const &grid() const { return grid_; }
	T *data() { return data_.data(); }
	T const *data() const { return data_.data(); }

	size_t bytes() const { return data_.size() * sizeof(T); }

	// copy/move operations
	Lattice(Lattice const &other)
	    : grid_(other.grid()),
	      data_(util::make_aligned_unique_span<T>(grid().size()))
	{
		++latticeAllocCount;
		parallel_memcpy(data(), other.data(), grid().size() * sizeof(T));
	}

	Lattice(Lattice &&other) noexcept
	    : grid_(std::exchange(other.grid_, {})),
	      data_(std::exchange(other.data_, {}))
	{}

	Lattice &operator=(Lattice const &other)
	{
		if (data_.size() != other.data_.size())
		{
			++latticeAllocCount;
			data_ = util::make_aligned_unique_span<T>(other.grid().size());
		}
		grid_ = other.grid();

		parallel_memcpy(data(), other.data(), grid().size() * sizeof(T));

		return *this;
	}

	Lattice &operator=(Lattice &&other) noexcept
	{
		grid_ = std::exchange(other.grid_, {});
		data_ = std::exchange(other.data_, {});
		return *this;
	}

	// single-element access. Slow, but sometimes nice to have
	T peek_site(Coordinate const &index) const
	{
		return data()[grid().flat_index(index)];
	}
	void poke_site(Coordinate const &index, T const &a)
	{
		data()[grid().flat_index(index)] = a;
	}

	// convert to ndarray. slow, pretty much only for debugging
	template <size_t N> util::ndarray<T, N> to_array() const
	{
		assert(N == grid().ndim());
		if constexpr (N == 2)
		{
			auto r = util::ndarray<T, N>(
			    {(size_t)grid().shape(0), (size_t)grid().shape(1)});
			for (int x = 0; x < grid().shape(0); ++x)
				for (int y = 0; y < grid().shape(1); ++y)
					r(x, y) = peek_site({x, y});
			return r;
		}
		else
			assert(false);
	}
};

// equivalent to Lattice<T>[Nd] width Nd = grid.ndim
//     * intended for a single "outer" (Lorentz) index
//     * all contained lattices are expected to live on the same grid
//          (not fully enforces sadly)
//     * better operator overloading than std::vector<Lattice>
//     * TODO: probably should integrate this functionality into the Lattice
//             class itself. Though keep the data-layout of Lorentz index
//             outside of the lattice index.
template <typename T> class LatticeStack
{
	std::vector<Lattice<T>> data_;

  public:
	LatticeStack() noexcept : LatticeStack(Grid{})
	{
		// the default grid has one dimension (of size 0), so we need to
		// create one (size 0) lattice here for consistency.
		// TODO: use util::{small,static}_vector to avoid this
	}

	explicit LatticeStack(Grid const &g)
	{
		data_.reserve(g.ndim());
		for (int i = 0; i < g.ndim(); ++i)
			data_.push_back(Lattice<T>(g));
	}

	Grid const &grid() const
	{
		assert(!data_.empty()); // should be allow 0-dimensional grids?

		Grid const &g = data_[0].grid();
		assert(g.ndim() == (int)data_.size());
		for (int i = 1; i < (int)data_.size(); ++i)
			assert(g == data_[i].grid());
		return g;
	}

	size_t size() const { return data_.size(); }
	Lattice<T> &operator[](size_t i) { return data_.at(i); }
	Lattice<T> const &operator[](size_t i) const { return data_.at(i); }
};

template <class T> struct is_lattice : std::integral_constant<bool, false>
{};
template <class T>
struct is_lattice<Lattice<T>> : std::integral_constant<bool, true>
{};
template <class T>
struct is_lattice<LatticeStack<T>> : std::integral_constant<bool, true>
{};

template <class T> inline constexpr bool is_lattice_v = is_lattice<T>::value;

/*
// store LatticeStack to file as [Nd,X,Y,Z,T,ncomp] array
//     * returns resulting dataset
template <typename vT>
util::Hdf5Dataset writeToFile(util::Hdf5File &file, std::string const &name,
                              LatticeStack<vT> const &a)
{
    auto &g = Grid::make(a.grid().shape(), 1);
    using Object = typename LatticeStack<vT>::Object;
    using Real = typename LatticeStack<vT>::Real;
    auto tmp = Lattice<Object>(g);

    // type and shape of resulting HDF5 dataset
    auto type = util::h5_type_id<Real>();
    std::vector<hsize_t> shape;
    shape.push_back(g.ndim());
    for (auto s : g.shape())
        shape.push_back(s);
    shape.push_back(vT::size());

    // create dataset and fill it
    auto dset = file.create_data(name, shape, type);
    for (size_t i = 0; i < a.size(); ++i)
    {
        a[i].transferTo(tmp);
        dset.write((hsize_t)i, tmp.rawSpanConst());
    }

    return dset;
}

template <typename vT>
void readFromFile(util::Hdf5File &file, std::string const &name,
                  LatticeStack<vT> &a)
{
    auto &g = Grid::make(a.grid().shape(), 1);
    using Object = typename LatticeStack<vT>::Object;
    auto tmp = Lattice<Object>(g);

    auto dset = file.open_data(name);
    // TODO: check shape more carefully
    for (size_t i = 0; i < a.size(); ++i)
    {
        dset.read((hsize_t)i, tmp.rawSpan());
        tmp.transferTo(a[i]);
    }
}
*/

template <typename F, typename T, typename... Ts>
void lattice_apply(F f, Lattice<T> &a, Lattice<Ts> const &...as)
{
	assert(((a.grid() == as.grid()) && ...));
	size_t size = a.grid().size();

#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < size; ++i)
		f(a.data()[i], as.data()[i]...);
}

template <typename F, typename T, typename... Ts>
void lattice_apply(F f, LatticeStack<T> &a, LatticeStack<Ts> const &...as)
{
	assert(((a.size() == as.size()) && ...));
	for (size_t i = 0; i < a.size(); ++i)
		lattice_apply(f, a[i], as[i]...);
}

template <typename F, typename T, typename... Ts>
auto lattice_sum(F f, Lattice<T> const &a, Lattice<Ts> const &...as)
    -> decltype(f(a.data()[0], as.data()[0]...))
{
	assert(((a.grid() == as.grid()) && ...));

	using R = decltype(f(a.data()[0], as.data()[0]...));
	auto r = R(0);

	// number of local accumulators:
	//   1) improve numerics (similar to "pair summation")
	//   2) improve instruction-level parallelism
	//   3) increases register pressure, so we scale it by the size of T
	//   4) the number '64' is kinda arbitrary
	constexpr size_t naccs =
	    std::max(size_t(1), std::bit_floor(size_t(64) / sizeof(T)));

	size_t size = a.grid().size();
	assert(size % naccs == 0);

#pragma omp parallel
	{
		R accs[naccs];
		for (size_t k = 0; k < naccs; ++k)
			accs[k] = R(0);

#pragma omp for schedule(static) nowait
		for (size_t i = 0; i < size / naccs; ++i)
			for (size_t k = 0; k < naccs; ++k)
				accs[k] +=
				    f(a.data()[i * naccs + k], as.data()[i * naccs + k]...);

		for (size_t k = 1; k < naccs; ++k)
			accs[0] += accs[k];

#pragma omp critical
		r += accs[0];
	}
	return r;
}

template <typename F, typename T, typename... Ts>
auto lattice_sum(F f, LatticeStack<T> const &a, LatticeStack<Ts> const &...as)
    -> decltype(f(a[0].data()[0], as[0].data()[0]...))
{
	auto r = decltype(f(a[0].data()[0], as[0].data()[0]...))(0);
	for (size_t i = 0; i < a.size(); ++i)
		r += lattice_sum(f, a[i], as[i]...);
	return r;
}

#define UTIL_DEFINE_LATTICE_BINARY(op)                                         \
	template <typename T>                                                      \
	Lattice<T> operator op(Lattice<T> const &a, Lattice<T> const &b)           \
	{                                                                          \
		assert(a.grid() == b.grid());                                          \
		auto r = Lattice<T>(a.grid());                                         \
		lattice_apply([](T &rr, T const &aa, T const &bb) { rr = aa op bb; },  \
		              r, a, b);                                                \
		return r;                                                              \
	}                                                                          \
	template <typename T>                                                      \
	Lattice<T> operator op(Lattice<T> &&a, Lattice<T> const &b)                \
	{                                                                          \
		assert(a.grid() == b.grid());                                          \
		lattice_apply([](T &aa, T const &bb) { aa op## = bb; }, a, b);         \
		return std::move(a);                                                   \
	}                                                                          \
	template <typename T>                                                      \
	Lattice<T> operator op(Lattice<T> const &a, Lattice<T> &&b)                \
	{                                                                          \
		assert(a.grid() == b.grid());                                          \
		lattice_apply([](T &bb, T const &aa) { bb = aa op bb; }, b, a);        \
		return std::move(b);                                                   \
	}                                                                          \
	template <typename T>                                                      \
	Lattice<T> operator op(Lattice<T> &&a, Lattice<T> &&b)                     \
	{                                                                          \
		return operator op(std::move(a), b);                                   \
	}                                                                          \
	template <typename T>                                                      \
	LatticeStack<T> operator op(LatticeStack<T> const &a,                      \
	                            LatticeStack<T> const &b)                      \
	{                                                                          \
		assert(a.size() == b.size());                                          \
		auto c = LatticeStack<T>(a.grid());                                    \
		lattice_apply([](T &cc, T const &aa, T const &bb) { cc = aa op bb; },  \
		              c, a, b);                                                \
		return c;                                                              \
	}                                                                          \
	template <typename T, typename U>                                          \
	Lattice<T> &operator op##=(Lattice<T> &a, Lattice<U> const &b)             \
	{                                                                          \
		assert(a.grid() == b.grid());                                          \
		lattice_apply([](T &aa, T const &bb) { aa op## = bb; }, a, b);         \
		return a;                                                              \
	}                                                                          \
	template <typename T, typename U>                                          \
	auto operator op(Lattice<T> const &a, U const &b)->Lattice<T>              \
	{                                                                          \
		auto r = Lattice<T>(a.grid());                                         \
		lattice_apply([b](T &rr, T const &aa) { rr = aa op b; }, r, a);        \
		return r;                                                              \
	}                                                                          \
	template <typename T, typename U>                                          \
	auto operator op(Lattice<T> &&a, U const &b)->Lattice<T>                   \
	{                                                                          \
		lattice_apply([b](T &aa) { aa = aa op b; }, a);                        \
		return std::move(a);                                                   \
	}                                                                          \
	template <typename T, typename U>                                          \
	Lattice<T> &operator op##=(Lattice<T> &a, U const &b)                      \
	{                                                                          \
		lattice_apply([b](T &aa) { aa op## = b; }, a);                         \
		return a;                                                              \
	}

#define UTIL_DEFINE_LATTICE_UNARY(name, fun)                                   \
	template <typename T>                                                      \
	auto name(Lattice<T> const &a)->Lattice<decltype(fun(std::declval<T>()))>  \
	{                                                                          \
		auto r = Lattice<decltype(fun(std::declval<T>()))>(a.grid());          \
		lattice_apply([](T &rr, T const &aa) { rr = fun(aa); }, r, a);         \
		return r;                                                              \
	}                                                                          \
	template <typename T> Lattice<T> name(Lattice<T> &&a)                      \
	{                                                                          \
		lattice_apply([](T &aa) { aa = fun(aa); }, a);                         \
		return std::move(a);                                                   \
	}                                                                          \
	template <typename T>                                                      \
	auto name(LatticeStack<T> const &a)                                        \
	    ->LatticeStack<decltype(fun(std::declval<T>()))>                       \
	{                                                                          \
		auto r = LatticeStack<decltype(fun(std::declval<T>()))>(a.grid());     \
		lattice_apply([](T &rr, T const &aa) { rr = fun(aa); }, r, a);         \
		return r;                                                              \
	}                                                                          \
	template <typename T> LatticeStack<T> name(LatticeStack<T> &&a)            \
	{                                                                          \
		lattice_apply([](T &aa) { aa = fun(aa); }, a);                         \
		return std::move(a);                                                   \
	}

#define UTIL_DEFINE_LATTICE_REDUCTION(name, fun)                               \
	template <typename T>                                                      \
	auto name(Lattice<T> const &a)->decltype(fun(std::declval<T>()))           \
	{                                                                          \
		return lattice_sum([](T const &aa) { return fun(aa); }, a);            \
	}                                                                          \
	template <typename T>                                                      \
	auto name(LatticeStack<T> const &a)->decltype(fun(std::declval<T>()))      \
	{                                                                          \
		return lattice_sum([](T const &aa) { return fun(aa); }, a);            \
	}

UTIL_DEFINE_LATTICE_BINARY(+)
UTIL_DEFINE_LATTICE_BINARY(-)
UTIL_DEFINE_LATTICE_BINARY(*)

struct my_identity
{
	template <class T> T operator()(T const &x) { return x; }
};
UTIL_DEFINE_LATTICE_REDUCTION(sum, my_identity{})

// keep these macros. used in gauge/utils.h
// #undef UTIL_DEFINE_LATTICE_BINARY
// #undef UTIL_DEFINE_LATTICE_UNARY
// #undef UTIL_DEFINE_LATTICE_REDUCTION

// TODO: this should be handled by templates like the operators above...
template <typename T> Lattice<T> operator*(Lattice<T> const &a, double b)
{
	auto r = Lattice<T>(a.grid());
	for (size_t i = 0; i < a.grid().size(); ++i)
		r.data()[i] = a.data()[i] * b;
	return r;
}
template <typename T> Lattice<T> operator*(Lattice<T> &&a, double b)
{
	for (size_t i = 0; i < a.grid().size(); ++i)
		a.data()[i] *= b;
	return std::move(a);
}

// a += b*c
template <class A, class B, class C>
    requires(!is_lattice_v<B>)
void add_mul(Lattice<A> &a, B b, Lattice<C> const &c)
{
	lattice_apply([b](A &aa, C const &cc) { aa += b * cc; }, a, c);
}

template <typename T>
Lattice<T> cshift(Lattice<T> const &a, int dir, int offset)
{
	util::StopwatchGuard swg{swCshift};

	auto g = a.grid();
	assert(0 <= dir && dir < g.ndim());
	assert(std::abs(offset) < g.shape(dir) && offset != 0);

	// collapse dimensions faster/slower than dir
	int slow = 1, fast = 1;
	for (int i = 0; i < dir; ++i)
		slow *= g.shape(i);
	int s = g.shape(dir);
	for (int i = dir + 1; i < g.ndim(); ++i)
		fast *= g.shape(i);

	auto r = Lattice<T>(a.grid());

	for (int x = 0; x < slow; ++x)
	{
		T *rp = r.data() + x * s * fast;
		T const *ap = a.data() + x * s * fast;

		if (offset < 0)
		{
			memcpy(rp, ap + (s + offset) * fast, sizeof(T) * fast * (-offset));
			memcpy(rp + (-offset) * fast, ap, sizeof(T) * fast * (s + offset));
		}

		if (offset > 0)
		{
			memcpy(rp + (s - offset) * fast, ap, sizeof(T) * fast * offset);
			memcpy(rp, ap + offset * fast, sizeof(T) * fast * (s - offset));
		}
	}

	return r;
}

// Same as std::memcpy, put parallelized using OMP
inline void parallel_memcpy(void *dest, void const *src, size_t count)
{
#pragma omp parallel
	{
		int rank = omp_get_thread_num();
		int nranks = omp_get_num_threads();
		size_t chunk = (count / nranks) & -4096;

		void *my_dest = (char *)dest + rank * chunk;
		void const *my_src = (char const *)src + rank * chunk;

		size_t my_size =
		    (rank == nranks - 1) ? count - (nranks - 1) * chunk : chunk;

		std::memcpy(my_dest, my_src, my_size);
	}
}

// Same as std::memset, put parallelized using OMP
inline void parallel_memset(void *dest, int ch, std::size_t count)
{
#pragma omp parallel
	{
		int rank = omp_get_thread_num();
		int nranks = omp_get_num_threads();
		size_t chunk = (count / nranks) & -4096;

		void *my_dest = (char *)dest + rank * chunk;

		size_t my_size =
		    (rank == nranks - 1) ? count - (nranks - 1) * chunk : chunk;

		std::memset(my_dest, ch, my_size);
	}
}

} // namespace mesh
