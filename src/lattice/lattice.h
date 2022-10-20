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
#include <type_traits>

namespace mesh {

inline int64_t latticeAllocCount = 0;
inline util::Stopwatch swCshift;

template <typename vT> class Lattice
{
  public:
	// in Grid, these types are called
	//    scalar_type, vector_type, scalar_object, vector_object
	using Real = typename TensorTraits<vT>::Real;
	using vReal = typename TensorTraits<vT>::vReal;
	using Object = typename TensorTraits<vT>::Object;
	using vObject = typename TensorTraits<vT>::vObject;
	static constexpr size_t simd_width = TensorTraits<vT>::simd_width;

	// sanity checks on the types
	static_assert(std::is_same_v<vT, std::decay_t<vT>>);
	static_assert(std::is_trivially_copyable_v<Object>);
	static_assert(std::is_trivially_copyable_v<vObject>);
	static_assert(sizeof(vObject) == simd_width * sizeof(Object));
	// static_assert(sizeof(Object) == Object::size() * sizeof(Real));
	static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>);

  private:
	Grid const *grid_ = nullptr;           // never null
	util::unique_span<vObject> data_ = {}; // null if grid is default/empty

  public:
	Lattice() : grid_(&Grid::make({0}, simd_width)) {}
	explicit Lattice(Grid const &g) : grid_(&g)
	{
		assert(grid().isize() == simd_width);
		++latticeAllocCount;
		data_ = util::make_aligned_unique_span<vObject>(grid().osize());
	}

	static Lattice zeros(Grid const &g)
	{
		auto a = Lattice(g);
		std::memset(a.data(), 0, g.size() * sizeof(Object));
		return a;
	}

	static Lattice ones(Grid const &g)
	{
		auto a = Lattice(g);
		for (size_t i = 0; i < g.osize(); ++i)
			a.data()[i] = vObject::one();
		return a;
	}

	Grid const &grid() const { return *grid_; }
	vObject *data() { return data_.data(); }
	vObject const *data() const { return data_.data(); }

	std::span<Real> rawSpan()
	{
		return std::span(data()[0].data(), grid().size() * data()[0].size());
	}
	std::span<const Real> rawSpan() const
	{
		return std::span(data()[0].data(), grid().size() * data()[0].size());
	}
	std::span<const Real> rawSpanConst() const { return rawSpan(); }

	// copy/move operations
	Lattice(Lattice const &other)
	    : grid_(&other.grid()),
	      data_(util::make_aligned_unique_span<vObject>(grid().osize()))
	{
		++latticeAllocCount;
		std::memcpy(data(), other.data(), grid().osize() * sizeof(vObject));
	}
	Lattice(Lattice &&other) noexcept
	    : grid_(std::exchange(other.grid_, &Grid::make({0}, simd_width))),
	      data_(std::exchange(other.data_, {}))
	{}
	Lattice &operator=(Lattice const &other)
	{
		if (data_.size() != other.data_.size())
		{
			++latticeAllocCount;
			data_ = util::make_aligned_unique_span<vObject>(grid().osize());
		}
		grid_ = &other.grid();
		std::memcpy(data(), other.data(), grid().osize() * sizeof(vObject));
		return *this;
	}
	Lattice &operator=(Lattice &&other) noexcept
	{
		grid_ = std::exchange(other.grid_, &Grid::make({0}, simd_width));
		data_ = std::exchange(other.data_, {});
		return *this;
	}

	/** single-element access. Slow, but sometimes nice to have. */
	Object peekSite(Coordinate const &index) const
	{
		auto [oIndex, iIndex] = grid().flatIndex(index);
		return vextract(data()[oIndex], iIndex);
	}

	void pokeSite(Coordinate const &index, Object const &a)
	{
		auto [oIndex, iIndex] = grid().flatIndex(index);
		vinsert(data()[oIndex], iIndex, a);
	}

	/** convert to ndarray. slow, pretty much only for debugging */
	template <size_t N> util::ndarray<Object, N> toArray() const
	{
		assert(N == grid().ndim());
		if constexpr (N == 2)
		{
			auto r = util::ndarray<Object, N>(
			    {(size_t)grid().shape(0), (size_t)grid().shape(1)});
			for (int x = 0; x < grid().shape(0); ++x)
				for (int y = 0; y < grid().shape(1); ++y)
					r(x, y) = peekSite({x, y});
			return r;
		}
		else
			assert(false);
	}

	// copy to different lattice of the shape, changing simd-layout on the fly
	template <typename vU> void transferTo(Lattice<vU> &other) const
	{
		assert((void *)data() != (void *)other.data());
		assert(grid().shape() == other.grid().shape());
		static_assert(std::is_same_v<Object, typename Lattice<vU>::Object>);

		if (grid().ndim() == 4)
		{
			int nx = grid().shape(0);
			int ny = grid().shape(1);
			int nz = grid().shape(2);
			int nt = grid().shape(3);
			for (int x = 0; x < nx; ++x)
				for (int y = 0; y < ny; ++y)
					for (int z = 0; z < nz; ++z)
						for (int t = 0; t < nt; ++t)
							other.pokeSite({x, y, z, t},
							               peekSite({x, y, z, t}));
		}
		else
			assert(false);
	}
};

// equivalent to Lattice<T>[Nd] width Nd = grid.ndim
//     * intended for a single "outer" (Lorentz) index
//     * all contained lattices are expected to live on the same grid
//     * better operator overloading than std::vector<Lattice>
template <typename vT> class LatticeStack
{
  public:
	using Real = typename Lattice<vT>::Real;
	using vReal = typename Lattice<vT>::vReal;
	using Object = typename Lattice<vT>::Object;
	using vObject = typename Lattice<vT>::vObject;
	static constexpr size_t simd_width = Lattice<vT>::simd_width;

  private:
	std::vector<Lattice<vT>> data_;

  public:
	LatticeStack() = default;
	LatticeStack(Grid const &g)
	{
		data_.reserve(g.ndim());
		for (int i = 0; i < g.ndim(); ++i)
			data_.push_back(Lattice<vT>(g));
	}
	static LatticeStack zeros(Grid const &g)
	{
		LatticeStack r;
		r.data_.reserve(g.ndim());
		for (int i = 0; i < g.ndim(); ++i)
			r.data.push_back(Lattice<vT>::zeros(g));
		return r;
	}
	static LatticeStack ones(Grid const &g)
	{
		LatticeStack r;
		r.data_.reserve(g.ndim());
		for (int i = 0; i < g.ndim(); ++i)
			r.data.push_back(Lattice<vT>::ones(g));
		return r;
	}

	Grid const &grid() const
	{
		if (data_.empty())
		{
			// default-lattice is 0-dimensional
			return Lattice<vT>().grid();
		}
		else
		{
			Grid const &g = data_[0].grid();
			assert(g.ndim() == (int)data_.size());
			for (int i = 1; i < (int)data_.size(); ++i)
				assert(compatible(data_[0], data_[i]));
			return g;
		}
	}

	bool empty() const { return data_.empty(); }
	size_t size() const { return data_.size(); }
	Lattice<vT> &operator[](size_t i) { return data_.at(i); }
	Lattice<vT> const &operator[](size_t i) const { return data_.at(i); }
};

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

template <typename T, typename U>
bool compatible(Lattice<T> const &a, Lattice<U> const &b)
{
	return &a.grid() == &b.grid();
}

template <typename F, typename T, typename... Ts>
void lattice_apply(F f, Lattice<T> &a, Lattice<Ts> const &...as)
{
	assert((compatible(a, as) && ...));

	size_t osize = a.grid().osize();

	// TODO: in principle, this should be the only place in the code where we
	//       do explicit multithreadig using OMP. Though some careful tests
	//       are in order before activating it
	//#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < osize; ++i)
		f(a.data()[i], as.data()[i]...);
}

template <typename F, typename T, typename... Ts>
void lattice_apply(F f, LatticeStack<T> &a, LatticeStack<Ts> const &...as)
{
	assert(((a.size() == as.size()) && ...));
	for (size_t i = 0; i < a.size(); ++i)
		lattice_apply(f, a[i], as[i]...);
}

#define UTIL_DEFINE_LATTICE_BINARY(op)                                         \
	template <typename T>                                                      \
	Lattice<T> operator op(Lattice<T> const &a, Lattice<T> const &b)           \
	{                                                                          \
		assert(compatible(a, b));                                              \
		auto r = Lattice<T>(a.grid());                                         \
		lattice_apply([](T &rr, T const &aa, T const &bb) { rr = aa op bb; },  \
		              r, a, b);                                                \
		return r;                                                              \
	}                                                                          \
	template <typename T>                                                      \
	Lattice<T> operator op(Lattice<T> &&a, Lattice<T> const &b)                \
	{                                                                          \
		assert(compatible(a, b));                                              \
		lattice_apply([](T &aa, T const &bb) { aa op## = bb; }, a, b);         \
		return std::move(a);                                                   \
	}                                                                          \
	template <typename T>                                                      \
	Lattice<T> operator op(Lattice<T> const &a, Lattice<T> &&b)                \
	{                                                                          \
		assert(compatible(a, b));                                              \
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
		assert(compatible(a, b));                                              \
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

#define UTIL_DEFINE_LATTICE_UNARY(fun)                                         \
	template <typename T>                                                      \
	auto fun(Lattice<T> const &a)->Lattice<decltype(fun(std::declval<T>()))>   \
	{                                                                          \
		auto r = Lattice<decltype(fun(std::declval<T>()))>(a.grid());          \
		lattice_apply([](T &rr, T const &aa) { rr = fun(aa); }, r, a);         \
		return r;                                                              \
	}                                                                          \
	template <typename T> Lattice<T> fun(Lattice<T> &&a)                       \
	{                                                                          \
		lattice_apply([](T &aa) { aa = fun(aa); }, a);                         \
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
	auto name(LatticeStack<T> const &a)                                        \
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
	using base_type = typename Lattice<T>::vReal;

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
	osFast *= sizeof(T) / sizeof(base_type);

	// simd shuffle
	typename base_type::int_type mask;
	for (int x = 0; x < isSlow; ++x)
		for (int y = 0; y < is; ++y)
			for (int z = 0; z < isFast; ++z)
				mask[(x * is + y) * isFast + z] =
				    (x * is + (y + (offset < 0 ? -1 : 1) + is) % is) * isFast +
				    z;

	auto r = Lattice<T>(a.grid());

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
