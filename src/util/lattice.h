#ifndef UTIL_LATTICE_H
#define UTIL_LATTICE_H

#include <array>
#include <cassert>
#include <functional>
#include <type_traits>

#include "util/span.h"

/** non-owning multi-dimensional periodic view on regular data */
template <typename T, int N> class Lattice
{
	template <typename U, int M> friend class Lattice;

	T *data_;
	std::array<int, N> shape_;
	std::array<int, N> stride_;
	std::array<int, N> offset_;

	int index(std::array<int, N> pos) const
	{
		int k = 0;
		for (int i = 0; i < N; ++i)
		{
			// NOTE: This is performance critical. Do not use modulo!
			assert(0 <= pos[i] && pos[i] < shape_[i]);
			pos[i] += offset_[i];
			if (pos[i] >= shape_[i])
				pos[i] -= shape_[i];
			k += pos[i] * stride_[i];
		}
		return k;
	}

  public:
	/** default constructor */
	Lattice() : data_(nullptr)
	{
		shape_.fill(0);
		offset_.fill(0);
		stride_.fill(0);
	}

	/** view on contiguous data */
	Lattice(span<T> data_, std::array<int, N> shape_)
	    : data_(data_.data()), shape_(shape_)
	{
		offset_.fill(0);
		int s = 1;
		for (int i = N - 1; i >= 0; --i)
		{
			stride_[i] = s;
			s *= shape_[i];
		}
		assert((int)data_.size() == size());
	}

	/** convert to const view */
	operator Lattice<const T, N>() const
	{
		Lattice<const T, N> r;
		r.data_ = data_;
		r.shape_ = shape_;
		r.stride_ = stride_;
		r.offset_ = offset_;
		return r;
	}

	/** element type */
	typedef T type;

	/** number of dimensions */
	static constexpr int rank() { return N; }

	/** shape of the lattice */
	std::array<int, N> shape() const { return shape_; }

	/** number of elements */
	int size() const
	{
		int s = 1;
		for (int i = 0; i < N; ++i)
			s *= shape_[i];
		return s;
	}

	/** element access */
	T &operator()(std::array<int, N> pos) { return data_[index(pos)]; }
	const T &operator()(std::array<int, N> pos) const
	{
		return data_[index(pos)];
	}

	/** periodic shift in one dimension */
	Lattice shift(int i, int dist)
	{
		assert(0 <= i && i < N);
		Lattice l = *this;
		l.offset_[i] += dist;
		l.offset_[i] %= l.shape_[i];
		l.offset_[i] += l.shape_[i];
		l.offset_[i] %= l.shape_[i];
		return l;
	}

	/** fix one index */
	Lattice<T, N - 1> slice(int k, int pos)
	{
		pos = (pos + offset_[k]) % shape_[k];
		assert(0 <= pos && pos < shape_[k]);
		Lattice<T, N - 1> l;
		l.data_ = data_ + pos * stride_[k];
		for (int i = 0; i < k; ++i)
		{
			l.shape_[i] = shape_[i];
			l.stride_[i] = stride_[i];
			l.offset_[i] = offset_[i];
		}
		for (int i = k + 1; i < N; ++i)
		{
			l.shape_[i - 1] = shape_[i];
			l.stride_[i - 1] = stride_[i];
			l.offset_[i - 1] = offset_[i];
		}
		return l;
	}
};

template <typename F, int N, typename... T>
auto contract(F &&f, Lattice<T, N>... spans)
{
	if constexpr (N == 0)
		return f(spans({})...);
	else if constexpr (N == 1)
	{

		auto n = std::get<0>(std::tuple(spans...)).shape()[0];
		std::invoke_result_t<F, T...> r = 0;
		for (int i = 0; i < n; ++i)
			r += f(spans({i})...);
		return r;
	}
	else
	{
		auto n = std::get<0>(std::tuple(spans...)).shape()[0];
		std::invoke_result_t<F, T...> r = 0;
		for (int i = 0; i < n; ++i)
			r += contract(f, spans.slice(0, i)...);
		return r;
	}
}

template <typename F, int N, typename T, typename... U>
void eval(Lattice<T, N> lhs, F &&f, Lattice<U, N>... spans)
{
	if constexpr (N == 0)
		lhs({}) = f(spans({})...);
	else if constexpr (N == 1)
	{
		auto n = std::get<0>(std::tuple(spans...)).shape()[0];
		for (int i = 0; i < n; ++i)
			lhs({i}) = f(spans({i})...);
	}
	else
	{
		auto n = std::get<0>(std::tuple(spans...)).shape()[0];
		for (int i = 0; i < n; ++i)
			eval(lhs.slice(0, i), f, spans.slice(0, i)...);
	}
}

#endif
