#pragma once

#include "fmt/format.h"
#include "lattice/grid.h"
#include "lattice/tensor.h"
#include "util/ndarray.h"
#include "util/span.h"
#include "util/stopwatch.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <type_traits>

namespace mesh {

inline int64_t latticeAllocCount = 0;
inline util::Stopwatch swCshift;

template <typename vT> class Lattice
{
  public:
	static constexpr size_t simdWidth = TensorTraits<vT>::simdWidth;
	using scalar_type = typename TensorTraits<vT>::ScalarType;
	using vector_type = vT;
	using base_type = typename TensorTraits<vT>::BaseType;

	// sanity checks on the types
	static_assert(std::is_same_v<vT, std::decay_t<vT>>);
	static_assert(std::is_trivially_copyable_v<scalar_type>);
	static_assert(std::is_trivially_copyable_v<vector_type>);
	static_assert(sizeof(vector_type) == simdWidth * sizeof(scalar_type));

  private:
	Grid const *grid_ = nullptr;  // never null
	vector_type *data_ = nullptr; // can be null if grid is the empty lattice

  public:
	Lattice() : grid_(&Grid::make({0}, simdWidth)) {}
	explicit Lattice(Grid const &g) : grid_(&g)
	{
		assert(grid().isize() == simdWidth);
		++latticeAllocCount;
		data_ = new vector_type[grid().osize()];
	}

	static Lattice zeros(Grid const &g)
	{
		auto a = Lattice(g);
		std::memset(a.data(), 0, g.size() * sizeof(scalar_type));
		return a;
	}

	static Lattice ones(Grid const &g)
	{
		auto a = Lattice(g);
		for (size_t i = 0; i < g.osize(); ++i)
			a.data()[i] = vector_type(scalar_type(1.0));
		return a;
	}

	Grid const &grid() const { return *grid_; }
	vector_type *data() { return data_; }
	vector_type const *data() const { return data_; }

	// copy/move operations
	Lattice(Lattice const &other)
	    : grid_(&other.grid()), data_(new vector_type[grid().osize()])
	{
		++latticeAllocCount;
		std::memcpy(data(), other.data(), grid().osize() * sizeof(vector_type));
	}
	Lattice(Lattice &&other) : grid_{&other.grid()}, data_{other.data()}
	{
		other.grid_ = &Grid::make({0}, simdWidth);
		other.data_ = nullptr;
	}
	Lattice &operator=(Lattice const &other)
	{
		if (grid().size() != other.grid().size())
		{
			delete[] data_;
			++latticeAllocCount;
			data_ = new vector_type[other.grid().osize()];
		}
		grid_ = &other.grid();
		std::memcpy(data(), other.data(), grid().osize() * sizeof(vector_type));
		return *this;
	}
	Lattice &operator=(Lattice &&other)
	{
		std::swap(grid_, other.grid_);
		std::swap(data_, other.data_);
		return *this;
	}

	~Lattice() { delete[] data_; }

	/** single-element access. Slow, but sometimes nice to have. */
	scalar_type peekSite(Coordinate const &index) const
	{
		auto [oIndex, iIndex] = grid().flatIndex(index);
		return vextract(data()[oIndex], iIndex);
	}

	void pokeSite(Coordinate const &index, scalar_type const &a)
	{
		auto [oIndex, iIndex] = grid().flatIndex(index);
		vinsert(data()[oIndex], iIndex, a);
	}

	/** convert to ndarray. slow, pretty much only for debugging */
	template <size_t N> util::ndarray<scalar_type, N> toArray() const
	{
		assert(N == grid().ndim());
		if constexpr (N == 2)
		{
			auto r = util::ndarray<scalar_type, N>(
			    {(size_t)grid().shape(0), (size_t)grid().shape(1)});
			for (int x = 0; x < grid().shape(0); ++x)
				for (int y = 0; y < grid().shape(1); ++y)
					r(x, y) = peekSite({x, y});
			return r;
		}
		else
			assert(false);
	}
};

template <typename T, typename U>
bool compatible(Lattice<T> const &a, Lattice<U> const &b)
{
	return &a.grid() == &b.grid();
}

template <typename F, typename T, typename... Ts>
void lattice_apply(F f, Lattice<T> &a, Lattice<Ts> const &... as)
{
	std::array<Grid const *, sizeof...(Ts)> grids = {&as.grid()...};
	for (size_t i = 0; i < sizeof...(Ts); ++i)
		assert(grids[i] == &a.grid());

	size_t osize = a.grid().osize();
	for (size_t i = 0; i < osize; ++i)
		f(a.data()[i], as.data()[i]...);
}

#define UTIL_DEFINE_LATTICE_BINARY(op)                                         \
	template <typename T>                                                      \
	Lattice<T> operator op(Lattice<T> const &a, Lattice<T> const &b)           \
	{                                                                          \
		assert(compatible(a, b));                                              \
		auto r = Lattice<T>(a.grid());                                         \
		for (size_t i = 0; i < a.grid().osize(); ++i)                          \
			r.data()[i] = a.data()[i] op b.data()[i];                          \
		return r;                                                              \
	}                                                                          \
	template <typename T>                                                      \
	Lattice<T> operator op(Lattice<T> &&a, Lattice<T> const &b)                \
	{                                                                          \
		assert(compatible(a, b));                                              \
		for (size_t i = 0; i < a.grid().osize(); ++i)                          \
			a.data()[i] op## = b.data()[i];                                    \
		return std::move(a);                                                   \
	}                                                                          \
	template <typename T>                                                      \
	Lattice<T> operator op(Lattice<T> const &a, Lattice<T> &&b)                \
	{                                                                          \
		assert(compatible(a, b));                                              \
		for (size_t i = 0; i < a.grid().osize(); ++i)                          \
			b.data()[i] = a.data()[i] op b.data()[i];                          \
		return std::move(b);                                                   \
	}                                                                          \
	template <typename T>                                                      \
	Lattice<T> operator op(Lattice<T> &&a, Lattice<T> &&b)                     \
	{                                                                          \
		return operator op(std::move(a), b);                                   \
	}                                                                          \
	template <typename T>                                                      \
	std::vector<Lattice<T>> operator op(std::vector<Lattice<T>> const &a,      \
	                                    std::vector<Lattice<T>> const &b)      \
	{                                                                          \
		assert(a.size() == b.size());                                          \
		auto c = std::vector<Lattice<T>>(a.size());                            \
		for (size_t i = 0; i < c.size(); ++i)                                  \
			c[i] = a[i] op b[i];                                               \
		return c;                                                              \
	}                                                                          \
	template <typename T, typename U>                                          \
	Lattice<T> &operator op##=(Lattice<T> &a, Lattice<U> const &b)             \
	{                                                                          \
		assert(compatible(a, b));                                              \
		for (size_t i = 0; i < a.grid().osize(); ++i)                          \
			a.data()[i] op## = b.data()[i];                                    \
		return a;                                                              \
	}                                                                          \
	template <typename T, typename U>                                          \
	auto operator op(Lattice<T> const &a, U const &b)->Lattice<T>              \
	{                                                                          \
		auto r = Lattice<T>(a.grid());                                         \
		for (size_t i = 0; i < a.grid().osize(); ++i)                          \
			r.data()[i] = a.data()[i] op b;                                    \
		return r;                                                              \
	}                                                                          \
	template <typename T, typename U>                                          \
	auto operator op(Lattice<T> &&a, U const &b)->Lattice<T>                   \
	{                                                                          \
		for (size_t i = 0; i < a.grid().osize(); ++i)                          \
			a.data()[i] = a.data()[i] op b;                                    \
		return std::move(a);                                                   \
	}                                                                          \
	template <typename T, typename U>                                          \
	Lattice<T> &operator op##=(Lattice<T> &a, U const &b)                      \
	{                                                                          \
		for (size_t i = 0; i < a.grid().osize(); ++i)                          \
			a.data()[i] op## = b;                                              \
		return a;                                                              \
	}

#define UTIL_DEFINE_LATTICE_UNARY(fun)                                         \
	template <typename T>                                                      \
	auto fun(Lattice<T> const &a)->Lattice<decltype(fun(std::declval<T>()))>   \
	{                                                                          \
		auto r = Lattice<decltype(fun(std::declval<T>()))>(a.grid());          \
		for (size_t i = 0; i < a.grid().osize(); ++i)                          \
			r.data()[i] = fun(a.data()[i]);                                    \
		return r;                                                              \
	}                                                                          \
	template <typename T> Lattice<T> fun(Lattice<T> &&a)                       \
	{                                                                          \
		for (size_t i = 0; i < a.grid().osize(); ++i)                          \
			a.data()[i] = fun(a.data()[i]);                                    \
		return std::move(a);                                                   \
	}

#define UTIL_DEFINE_LATTICE_REDUCTION(name, fun)                               \
	template <typename T>                                                      \
	auto name(Lattice<T> const &a)->decltype(vsum(fun(std::declval<T>())))     \
	{                                                                          \
		decltype(fun(std::declval<T>())) s = {};                               \
		for (size_t i = 0; i < a.grid().osize(); ++i)                          \
			s += fun(a.data()[i]);                                             \
		return vsum(s);                                                        \
	}                                                                          \
	template <typename T>                                                      \
	auto name(std::vector<Lattice<T>> const &a)                                \
	    ->decltype(vsum(fun(std::declval<T>())))                               \
	{                                                                          \
		decltype(vsum(fun(std::declval<T>()))) s = {};                         \
		for (size_t i = 0; i < a.size(); ++i)                                  \
			s += name(a[i]);                                                   \
		return s;                                                              \
	}

UTIL_DEFINE_LATTICE_BINARY(+)
UTIL_DEFINE_LATTICE_BINARY(-)
UTIL_DEFINE_LATTICE_BINARY(*)

UTIL_DEFINE_LATTICE_UNARY(trace)
UTIL_DEFINE_LATTICE_UNARY(adj)
UTIL_DEFINE_LATTICE_UNARY(exp)
UTIL_DEFINE_LATTICE_UNARY(projectOnAlgebra)

UTIL_DEFINE_LATTICE_REDUCTION(sum, sum)
UTIL_DEFINE_LATTICE_REDUCTION(norm2, norm2)
UTIL_DEFINE_LATTICE_REDUCTION(sumTrace, trace)

#undef UTIL_DEFINE_LATTICE_BINARY
#undef UTIL_DEFINE_LATTICE_UNARY
#undef UTIL_DEFINE_LATTICE_REDUCTION

// TODO: this should be handled by templates like the operators above...
template <typename T> Lattice<T> operator*(Lattice<T> const &a, double b)
{
	auto r = Lattice<T>(a.grid());
	for (size_t i = 0; i < a.grid().osize(); ++i)
		r.data()[i] = a.data()[i] * b;
	return r;
}
template <typename T> Lattice<T> operator*(Lattice<T> &&a, double b)
{
	for (size_t i = 0; i < a.grid().osize(); ++i)
		a.data()[i] *= b;
	return std::move(a);
}

template <typename T>
Lattice<T> cshift(Lattice<T> const &a, int dir, int offset)
{
	util::StopwatchGuard swg{swCshift};

	auto const &g = a.grid();
	assert(0 <= dir && dir < g.ndim());
	assert(abs(offset) < g.oshape(1));

	// collapse dimensions faster/slower than dir
	int osSlow = 1, isSlow = 1;
	for (int i = 0; i < dir; ++i)
	{
		osSlow *= g.oshape(i);
		isSlow *= g.ishape(i);
	}
	int os = g.oshape(dir);
	int is = g.ishape(dir);
	int osFast = 1, isFast = 1;
	for (int i = dir + 1; i < g.ndim(); ++i)
	{
		osFast *= g.oshape(i);
		isFast *= g.ishape(i);
	}
	osFast *= sizeof(T) / sizeof(typename Lattice<T>::base_type);

	// simd shuffle
	typename Lattice<T>::base_type::int_type mask;
	for (int x = 0; x < isSlow; ++x)
		for (int y = 0; y < is; ++y)
			for (int z = 0; z < isFast; ++z)
				mask[(x * is + y) * isFast + z] =
				    (x * is + (y + (offset < 0 ? -1 : 1) + is) % is) * isFast +
				    z;

	auto r = Lattice<T>(a.grid());

	using base_type = typename Lattice<T>::base_type;
	base_type *rp;
	base_type const *ap;

	// non-wrapped part
	for (int x = 0; x < osSlow; ++x)
	{

		if (offset >= 0)
		{
			rp = (base_type *)r.data() + x * os * osFast;
			ap = (base_type const *)a.data() + (x * os + offset) * osFast;
		}
		else
		{
			rp = (base_type *)r.data() + (x * os - offset) * osFast;
			ap = (base_type const *)a.data() + x * os * osFast;
		}

		std::memcpy(rp, ap, sizeof(base_type) * osFast * (os - abs(offset)));
	}

	// wrapped part
	for (int x = 0; x < osSlow; ++x)
	{
		if (offset >= 0)
		{
			rp = (base_type *)r.data() + (x * os - offset + os) * osFast;
			ap = (base_type const *)a.data() + x * os * osFast;
		}
		else // offset < 0
		{
			rp = (base_type *)r.data() + (x * os) * osFast;
			ap =
			    (base_type const *)a.data() + (x * os + (offset + os)) * osFast;
		}

		// NOTE: Right here, we could also implement other boundary conditions.
		//       For example multiplying some lanes by -1 for anti-periodic.
		for (int t = 0; t < osFast * abs(offset); ++t)
			rp[t] = vshuffle(ap[t], mask);
	}

	return r;
}

} // namespace mesh
