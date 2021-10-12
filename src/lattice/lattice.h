#pragma once

#include "fmt/format.h"
#include "util/ndarray.h"
#include "util/span.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <type_traits>

namespace mesh {

template <typename T, size_t N>
util::ndarray<T, N> cshift(util::ndarray<T, N> const &a, int dir, int offset)
{
	size_t L = a.shape(dir);
	auto b = util::ndarray<T, N>{a.shape()};
	if (offset >= 0)
	{
		b.slice(dir, 0, L - offset) = a.slice(dir, offset, L);
		b.slice(dir, L - offset, L) = a.slice(dir, 0, offset);
	}
	else
	{
		b.slice(dir, -offset, L) = a.slice(dir, 0, L + offset);
		b.slice(dir, 0, -offset) = a.slice(dir, L + offset, L);
	}
	return b;
}

template <typename T, size_t N>
util::ndarray<T, N> operator*(util::ndarray<T, N> const &a,
                              util::ndarray<T, N> const &b)
{
	assert(a.shape() == b.shape());
	auto r = util::ndarray<T, N>(a.shape());
	map([](T &rr, T const &aa, T const &bb) { rr = aa * bb; }, r, a, b);
	return r;
} // namespace mesh

template <typename T, size_t N>
util::ndarray<T, N> adj(util::ndarray<T, N> const &a)
{
	auto r = util::ndarray<T, N>(a.shape());
	map([](T &rr, T const &aa) { rr = adj(aa); }, r, a);
	return r;
}

template <typename T, size_t N>
util::ndarray<T, N> antiHermitianTraceless(util::ndarray<T, N> const &a)
{
	auto r = util::ndarray<T, N>(a.shape());
	map([](T &rr, T const &aa) { rr = antiHermitianTraceless(aa); }, r, a);
	return r;
}

template <typename T, size_t N>
util::ndarray<T, N> exp(util::ndarray<T, N> const &a)
{
	auto r = util::ndarray<T, N>(a.shape());
	map([](T &rr, T const &aa) { rr = exp(aa); }, r, a);
	return r;
}

template <typename T, size_t N>
util::ndarray<T, N> operator*(util::ndarray<T, N> const &a, double b)
{
	auto r = util::ndarray<T, N>(a.shape());
	map([&b](T &rr, T const &aa) { rr = aa * b; }, r, a);
	return r;
}

template <typename T, size_t N> T sum(util::ndarray<T, N> const &a)
{
	auto s = T(0);
	map([&s](T const &aa) { s += aa; }, a);
	return s;
}

template <typename T, size_t N> double norm2(util::ndarray<T, N> const &a)
{
	double s = 0;
	map([&s](T const &aa) { s += norm2(aa); }, a);
	return s;
}

} // namespace mesh
