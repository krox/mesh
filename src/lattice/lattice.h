#pragma once

#include "fmt/format.h"
#include "lattice/device.h"
#include "lattice/grid.h"
#include "lattice/tensor.h"
#include "util/complex.h"
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
	device_buffer<T> buffer_;

  public:
	Lattice() = default;

	// allocates memory but does not initialize anything
	explicit Lattice(Grid const &g)
	    : grid_(g), buffer_(device_allocate<T>(grid().size()))
	{
		++latticeAllocCount;
	}

	// allocates memory and set it to zero
	//   (zero bits, does not take the object type into account)
	static Lattice zeros(Grid const &g)
	{
		auto a = Lattice(g);
		a.fill_zeros();
		return a;
	}

	void fill_zeros() { device_clear(buffer_.get()); }
	// void fill(T const &a) { buffer().fill(a); }

	Grid const &grid() const { return grid_; }
	T *data() { return buffer_.get().data(); }
	T const *data() const { return buffer_.get().data(); }
	std::span<T> buffer() { return buffer_.get(); }
	std::span<const T> buffer() const { return buffer_.get(); }

	size_t size() const { return buffer().size(); }
	size_t bytes() const { return buffer().size() * sizeof(T); }

	// copy/move operations
	Lattice(Lattice const &other)
	    : grid_(other.grid()), buffer_(device_allocate<T>(other.size()))
	{
		device_copy(buffer(), other.buffer());
		++latticeAllocCount;
	}

	Lattice(Lattice &&other) noexcept
	    : grid_(std::exchange(other.grid_, {})),
	      buffer_(std::exchange(other.buffer_, {}))
	{}

	Lattice &operator=(Lattice const &other)
	{
		if (bytes() != other.bytes())
		{
			++latticeAllocCount;
			buffer_ = device_allocate<T>(other.size());
		}
		grid_ = other.grid();
		device_copy(buffer(), other.buffer());
		return *this;
	}

	Lattice &operator=(Lattice &&other) noexcept
	{
		grid_ = std::exchange(other.grid_, {});
		buffer_ = std::exchange(other.buffer_, {});
		return *this;
	}

	// copy from host memory
	void copy_from_host(std::span<const T> data)
	{
		assert(data.size() == size());
		device_copy_from_host(buffer(), data);
	}

	// copy to host memory
	std::vector<T> copy_to_host() const
	{
		std::vector<T> r(size());
		device_copy_to_host(r, buffer());
		return r;
	}

	// single-element access. Slow, but sometimes nice to have
	/*T peek_site(Coordinate const &index) const
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
	}*/
};

template <class T, class... Ts>
void conformable(Lattice<T> const &a, Lattice<Ts> const &...as)
{
	assert(((a.grid() == as.grid()) && ...));
}

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

// TODO: get rid of the 'lattice_apply' wrappers. using 'device_apply' directly
// is fine
template <typename F, typename T, typename... Ts>
void lattice_apply(F f, Lattice<T> &a, Lattice<Ts> const &...as)
{
	conformable(a, as...);
	device_apply(f, a.buffer(), as.buffer()...);
}

template <typename F, typename T, typename... Ts>
auto lattice_sum_apply(F f, Lattice<T> const &a, Lattice<Ts> const &...as)
{
	conformable(a, as...);
	return device_sum_apply(f, a.buffer(), as.buffer()...);
}

template <class T> T lattice_sum(Lattice<T> const &a)
{
	return device_sum(a.buffer());
}

// TODO: plenty of rvalue-reference overloads (after LatticeStack is gone)

#define MESH_DEFINE_LATTICE_UNARY(name, fun)                                   \
	template <class T>                                                         \
	auto name(Lattice<T> const &arg)                                           \
	    ->Lattice<decltype(fun(std::declval<T>()))>                            \
	{                                                                          \
		using R = decltype(fun(std::declval<T>()));                            \
		auto ret = Lattice<R>(arg.grid());                                     \
		lattice_apply([] UTIL_DEVICE(R &a, T const &b) { a = fun(b); }, ret,   \
		              arg);                                                    \
		return ret;                                                            \
	}

#define MESH_DEFINE_LATTICE_BINARY(op)                                         \
	template <class T, class U>                                                \
	auto operator op(Lattice<T> const &lhs, Lattice<U> const &rhs)             \
	    ->Lattice<decltype(std::declval<T>() op std::declval<U>())>            \
	{                                                                          \
		assert(lhs.grid() == rhs.grid());                                      \
		using R = decltype(std::declval<T>() op std::declval<U>());            \
		auto ret = Lattice<R>(lhs.grid());                                     \
		lattice_apply(                                                         \
		    [] UTIL_DEVICE(R &a, T const &b, U const &c) { a = b op c; }, ret, \
		    lhs, rhs);                                                         \
		return ret;                                                            \
	}                                                                          \
	template <class T, class U>                                                \
	Lattice<T> &operator op##=(Lattice<T> &lhs, Lattice<U> const &rhs)         \
	{                                                                          \
		assert(lhs.grid() == rhs.grid());                                      \
		lattice_apply([] UTIL_DEVICE(T &a, U const &b) { a op## = b; }, lhs,   \
		              rhs);                                                    \
		return lhs;                                                            \
	}                                                                          \
	template <class T, class U>                                                \
	    requires(!is_lattice_v<U>)                                             \
	auto operator op(Lattice<T> const &lhs, U const &rhs)                      \
	    ->Lattice<decltype(std::declval<T>() op std::declval<U>())>            \
	{                                                                          \
		using R = decltype(std::declval<T>() op std::declval<U>());            \
		auto ret = Lattice<R>(lhs.grid());                                     \
		lattice_apply([rhs] UTIL_DEVICE(R &a, T const &b) { a = b op rhs; },   \
		              ret, lhs);                                               \
		return ret;                                                            \
	}                                                                          \
	template <class T, class U>                                                \
	    requires(!is_lattice_v<U>)                                             \
	Lattice<T> &operator op##=(Lattice<T> &lhs, U const &rhs)                  \
	{                                                                          \
		lattice_apply([rhs] UTIL_DEVICE(T &a) { a op## = rhs; }, lhs);         \
		return lhs;                                                            \
	}

#define MESH_DEFINE_LATTICE_SUM(name, fun)                                     \
	template <class T>                                                         \
	auto name(Lattice<T> const &arg)->decltype(fun(std::declval<T>()))         \
	{                                                                          \
		return lattice_sum_apply(                                              \
		    [] UTIL_DEVICE(T const &a) { return fun(a); }, arg);               \
	}

MESH_DEFINE_LATTICE_BINARY(+)
MESH_DEFINE_LATTICE_BINARY(-)
MESH_DEFINE_LATTICE_BINARY(*)

// template <class T> T sum(Lattice<T> const &arg) { return lattice_sum(arg); }

// keep these macros. used in gauge/utils.h
// #undef MESH_DEFINE_LATTICE_BINARY
// #undef MESH_DEFINE_LATTICE_UNARY
// #undef MESH_DEFINE_LATTICE_REDUCTION

// TODO: this should be handled by templates like the operators above...
template <class T> Lattice<T> operator*(Lattice<T> const &lhs, double rhs)
{
	auto ret = Lattice<T>(lhs.grid());
	lattice_apply([rhs] UTIL_DEVICE(T & a, T const &b) { a = b * rhs; }, ret,
	              lhs);
	return ret;
}

// a += b*c
template <class A, class B, class C>
    requires(!is_lattice_v<B>)
void add_mul(Lattice<A> &lhs, B mhs, Lattice<C> const &rhs)
{
	lattice_apply([mhs] UTIL_DEVICE(A & a, C const &b) { a += mhs * b; }, lhs,
	              rhs);
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
	auto dst = util::span_2d(r.data(), slow, s * fast);
	auto src = util::span_2d(a.data(), slow, s * fast);
	assert(dst.size() == src.size() && dst.size() == a.size());
	device_cshift(dst, src, offset * fast);
	return r;
}

} // namespace mesh
