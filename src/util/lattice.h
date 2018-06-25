#ifndef UTIL_LATTICE_H
#define UTIL_LATTICE_H

#include <array>
#include <cassert>
#include <functional>
#include <type_traits>

#include "util/span.h"

/** this acts as range and iterator for lattice positions */
template <int rank> struct LatticeRange
{
	std::array<int, rank> shape_;
	std::array<int, rank> pos_;

  public:
	/** constructor */
	LatticeRange() = default;
	explicit LatticeRange(std::array<int, rank> shape_) : shape_(shape_)
	{
		pos_.fill(0);
	}
	LatticeRange(std::array<int, rank> shape_, std::array<int, rank> pos_)
	    : shape_(shape_), pos_(pos_)
	{}

	/** current position */
	std::array<int, rank> operator*() const { return pos_; }

	/* advance to next position */
	LatticeRange &operator++()
	{
		pos_[rank - 1] += 1;
		for (int i = rank - 1; i > 0; --i)
			if (pos_[i] >= shape_[i])
			{
				pos_[i] -= shape_[i];
				pos_[i - 1] += 1;
			}
			else
				break;
		return *this;
	}

	/** compare iterators */
	bool operator!=(const LatticeRange<rank> &b) const
	{
		assert(shape_ == b.shape_);
		for (int i = 0; i < rank; ++i)
			if (pos_[i] != b.pos_[i])
				return true;
		return false;
	}

	LatticeRange begin() const { return LatticeRange(shape_); }

	LatticeRange end() const
	{
		auto r = LatticeRange(shape_);
		r.pos_[0] = r.shape_[0];
		return r;
	}
};

template <typename T, int rank_> class Lattice;
template <typename L, typename U> class LatticeUnary;
template <typename La, typename Lb, typename U> class LatticeBinary;

template <typename L, typename T, int N> class BasicLattice
{
	// this prevents some wrong usage of CRTP pattern
	BasicLattice(){};
	friend L;

  public:
	/** helper for crtp: static cast to concrete type */
	typedef L Con;
	L &con() { return static_cast<L &>(*this); }
	const L &con() const { return static_cast<const L &>(*this); }

	/** element type */
	typedef T type;

	/** number of dimensions */
	static constexpr int rank() { return N; }

	/** number of elements */
	int size() const
	{
		int s = 1;
		for (int i = 0; i < rank(); ++i)
			s *= con().shape()[i];
		return s;
	}

	/** range/iterator over indices (not values) */
	LatticeRange<N> range() const { return LatticeRange<N>(con().shape()); }

	/** element-wise function application */
	template <typename F> auto map(F &&f) const
	{
		typedef std::invoke_result_t<F, const T &> U;
		return LatticeUnary<L, U>(con(), f);
	}

	/* element-wise multiplication */
	template <typename Lb> auto operator*(const Lb lattB) const
	{
		typedef typename Lb::type V;
		auto fun = [](const T &x, const V &y) { return x * y; };
		typedef std::invoke_result_t<decltype(fun), const T &, const V &> U;
		return LatticeBinary<L, typename Lb::Con, U>(con(), lattB.con(), fun);
	}

	T sum()
	{
		T s = 0;
		for (auto pos : range())
			s += con()(pos);
		return s;
	}

	Lattice<T, N> eval(span<T> buf) const
	{
		assert(buf.size() == (size_t)size());
		auto r = Lattice<T, N>(buf, con().shape());
		for (auto i : r.range())
			r(i) = con()(i);
		return r;
	}
};

/** non-owning multi-dimensional view on a contiguous array */
template <typename T, int rank_>
class Lattice : public BasicLattice<Lattice<T, rank_>, T, rank_>
{
	template <typename U, int r> friend class Lattice;

	T *data_;
	std::array<int, rank_> shape_;
	std::array<int, rank_> stride_;
	std::array<int, rank_> offset_;

	int index(std::array<int, rank_> pos) const
	{
		int k = 0;
		for (int i = 0; i < rank_; ++i)
		{
			pos[i] =
			    ((pos[i] + offset_[i]) % shape_[i] + shape_[i]) % shape_[i];
			k += pos[i] * stride_[i];
		}
		return k;
	}

  public:
	/** constructors */
	Lattice() : data_(nullptr)
	{
		shape_.fill(0);
		offset_.fill(0);
		stride_.fill(0);
	}

	Lattice(span<T> data_, std::array<int, rank_> shape_)
	    : data_(data_.data()), shape_(shape_)
	{
		offset_.fill(0);
		int s = 1;
		for (int i = rank_ - 1; i >= 0; --i)
		{
			stride_[i] = s;
			s *= shape_[i];
		}
		assert((int)data_.size() == this->size());
	}

	operator Lattice<const T, rank_>() const
	{
		Lattice<const T, rank_> r;
		r.data_ = data_;
		r.shape_ = shape_;
		r.stride_ = stride_;
		r.offset_ = offset_;
		return r;
	}

	/** shape of the lattice */
	std::array<int, rank_> shape() const { return shape_; }

	/** element access */
	T &operator()(std::array<int, rank_> pos) { return data_[index(pos)]; }
	const T &operator()(std::array<int, rank_> pos) const
	{
		return data_[index(pos)];
	}

	/** periodic shift in one dimension */
	Lattice shift(int i, int dist)
	{
		Lattice l = *this;
		l.offset_[i] += dist;
		return l;
	}

	/** fix one index */
	Lattice<T, rank_ - 1> slice(int k, int pos)
	{
		pos = (pos + offset_[k]) % shape_[k];
		assert(0 <= pos && pos < shape_[k]);
		Lattice<T, rank_ - 1> l;
		l.data_ = data_ + pos * stride_[k];
		for (int i = 0; i < k; ++i)
		{
			l.shape_[i] = shape_[i];
			l.stride_[i] = stride_[i];
			l.offset_[i] = offset_[i];
		}
		for (int i = k + 1; i < rank_; ++i)
		{
			l.shape_[i - 1] = shape_[i];
			l.stride_[i - 1] = stride_[i];
			l.offset_[i - 1] = offset_[i];
		}
		return l;
	}
};

/** element-wise unary expression */
template <typename L, typename U>
class LatticeUnary : public BasicLattice<LatticeUnary<L, U>, U, L::rank()>
{
	L latt;
	typedef std::function<U(const typename L::type &)> fun_t;
	fun_t fun;

  public:
	LatticeUnary() = default;
	LatticeUnary(L latt, fun_t fun) : latt(latt), fun(fun) {}

	std::array<int, L::rank()> shape() const { return latt.shape(); }

	U operator()(std::array<int, L::rank()> pos) const
	{
		return fun(latt(pos));
	}
};

/** element-wise binary expression */
template <typename La, typename Lb, typename U>
class LatticeBinary
    : public BasicLattice<LatticeBinary<La, Lb, U>, U, La::rank()>
{
	La lattA;
	Lb lattB;
	typedef std::function<U(const typename La::type &,
	                        const typename Lb::type &)>
	    fun_t;
	fun_t fun;

  public:
	LatticeBinary() = default;
	LatticeBinary(La lattA, Lb lattB, fun_t fun)
	    : lattA(lattA), lattB(lattB), fun(fun)
	{
		static_assert(La::rank() == Lb::rank());
		assert(lattA.shape() == lattB.shape());
	}

	std::array<int, La::rank()> shape() const { return lattA.shape(); }

	auto operator()(std::array<int, La::rank()> pos) const
	{
		return fun(lattA(pos), lattB(pos));
	}
};

#endif
