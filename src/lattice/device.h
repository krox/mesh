#pragma once

// This module handles some low-level management of device memory.
// Note that this header does not need to be compiled with nvcc.

#include "util/memory.h"
#include "util/span.h"
#include <cstddef>
#include <memory>
#include <type_traits>

namespace mesh {

// un-typed backend implementation
void *device_malloc(size_t n);
void device_free(void *ptr);
void device_memcpy(void *dst, void const *src, size_t n);
void device_memcpy_2d(void *dst, size_t dpitch, void const *src, size_t spitch,
                      size_t width, size_t height);
void device_memclear(void *ptr, size_t n);
void device_memcpy_to_host(void *dst, void const *src, size_t n);
void device_memcpy_from_host(void *dst, void const *src, size_t n);

void device_synchronize();

// RAII wrapper for device memory
template <class T> struct device_deleter
{
	void operator()(T *ptr, size_t) noexcept { device_free(ptr); }
};
template <class T>
using device_buffer = util::unique_span<T, device_deleter<T>>;

template <class T> device_buffer<T> device_allocate(size_t n)
{
	return device_buffer<T>(static_cast<T *>(device_malloc(n * sizeof(T))), n);
}

// typed memory movement for convenience
template <class T> void device_copy(std::span<T> dst, std::span<const T> src)
{
	assert(dst.size() == src.size());
	device_memcpy(dst.data(), src.data(), dst.size() * sizeof(T));
}
template <class T>
void device_copy(util::span_2d<T> dst, util::span_2d<const T> src)
{
	assert(dst.height() == src.height());
	assert(dst.width() == src.width());
	device_memcpy_2d(dst.data(), dst.stride() * sizeof(T), src.data(),
	                 src.stride() * sizeof(T), src.width() * sizeof(T),
	                 src.height());
}

// circular shift by 'offset' elements. Sign convention:
//    dst[i] = src[i + offset]
template <class T>
void device_cshift(std::span<T> dst, std::span<const T> src, int64_t offset)
{
	if (dst.data() == src.data())
		throw std::runtime_error("inplace cshift not supported");
	size_t n = dst.size();
	if (src.size() != n)
		throw std::runtime_error("device_cshift failed (size mismatch)");

	if (offset == 0)
	{
		device_copy(dst, src);
	}
	else if (offset > 0)
	{
		device_copy(dst.first(n - offset), src.last(n - offset));
		device_copy(dst.last(offset), src.first(offset));
	}
	else
	{
		device_copy(dst.last(n + offset), src.first(n + offset));
		device_copy(dst.first(-offset), src.last(-offset));
	}
}

// shift each row of a 2D array by 'offset' elements. Sign convention:
//    dst[i, j] = src[i, j + offset]
template <class T>
void device_cshift(util::span_2d<T> dst, util::span_2d<const T> src,
                   int64_t offset)
{
	if (dst.data() == src.data())
		throw std::runtime_error("inplace cshift(2d) not supported");
	if (dst.height() != src.height() || dst.width() != src.width())
		throw std::runtime_error("device_cshift(2d) failed (shape mismatch)");
	size_t n = dst.width();

	if (offset == 0)
	{
		device_copy(dst, src);
	}
	else if (offset > 0)
	{
		device_copy(dst.first_columns(n - offset),
		            src.last_columns(n - offset));
		device_copy(dst.last_columns(offset), src.first_columns(offset));
	}
	else
	{
		device_copy(dst.last_columns(n + offset),
		            src.first_columns(n + offset));
		device_copy(dst.first_columns(-offset), src.last_columns(-offset));
	}
}

// set memory to zero (NOTE: this is bit-wise zero, ignoring the type T)
template <class T> void device_clear(std::span<T> data)
{
	device_memclear(data.data(), data.size() * sizeof(T));
}

template <class T>
void device_copy_to_host(std::span<T> dst, std::span<const T> src)
{
	assert(dst.size() == src.size());
	device_memcpy_to_host(dst.data(), src.data(), dst.size() * sizeof(T));
}

template <class T> std::vector<T> device_copy_to_host(std::span<const T> src)
{
	std::vector<T> dst(src.size());
	device_copy_to_host(std::span(dst), src);
	return dst;
}

template <class T>
void device_copy_from_host(std::span<T> dst, std::span<const T> src)
{
	assert(dst.size() == src.size());
	device_memcpy_from_host(dst.data(), src.data(), dst.size() * sizeof(T));
}

#ifdef __CUDACC__

// apply a function to all elements of an array, multiple arrays in parallel.
//     * all arrays must be of same size
//     * pointers must be device pointers
//     * f should typically be a `__device__` lambda
template <class F, class... Ts>
__global__ void device_apply_kernel(F f, size_t size, Ts *...as)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;
	f(as[i]...);
}

template <class T>
__global__ void device_sum_kernel(size_t size, T *r, T const *a)
{
	// this would be more reasonable, but shared memory with a templated type is
	// not supported by nvcc
	// extern __shared__ R s[];
	extern __shared__ char s_raw[];
	T *s = (T *)(s_raw);

	size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	size_t local_index = threadIdx.x;

	if (global_index < size)
		s[local_index] = a[global_index];
	__syncthreads();

	// reduction in shared memory. pair-summation should give good performance
	// and typically stable numerics as well.
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (local_index < stride && global_index + stride < size)
			s[local_index] += s[local_index + stride];
		__syncthreads();
	}

	if (local_index == 0)
		r[blockIdx.x] = s[0];
}

// produces one value per thread-block. Size of thread-block should be a power
// of two.
template <class F, class R, class... Ts>
__global__ void device_sum_apply_kernel(F f, size_t size, R *r, Ts *...as)
{
	// this would be more reasonable, but shared memory with a templated type is
	// not supported by nvcc
	// extern __shared__ R s[];
	extern __shared__ char s_raw[];
	R *s = (R *)(s_raw);

	size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	size_t local_index = threadIdx.x;

	if (global_index < size)
		s[local_index] = f(as[global_index]...);
	__syncthreads();

	// reduction in shared memory. pair-summation should give good performance
	// and typically stable numerics as well.
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		if (local_index < stride && global_index + stride < size)
			s[local_index] += s[local_index + stride];
		__syncthreads();
	}

	if (local_index == 0)
		r[blockIdx.x] = s[0];
}

// Apply a function to all elements of a lattice, multiple lattices in parallel.
// All lattices must be of same size.
// typical usage:
//   device_apply( [=alpha](double&a, double const& b) UTIL_DEVICE
//                { a += alpha * b; }, A, B);
template <class F, class T, class... Ts>
void device_apply(F f, std::span<T> a, std::span<const Ts>... as)
{
	assert(((a.size() == as.size()) && ...));

	constexpr size_t block_size = 256; // tunable in [32,1024]
	size_t nblocks = (a.size() + block_size - 1) / block_size;
	device_apply_kernel<<<nblocks, block_size>>>(f, a.size(), a.data(),
	                                             as.data()...);
}

template <class T> auto device_sum(std::span<const T> a) -> T
{
	constexpr size_t block_size = 256;
	size_t nblocks = (a.size() + block_size - 1) / block_size;
	auto r = device_allocate<T>(nblocks);

	device_sum_kernel<<<nblocks, block_size, block_size * sizeof(T)>>>(
	    a.size(), r.data(), a.data());
	device_synchronize(); // is this necessary?
	auto r_host = r.copy_to_host();
	auto result = r_host[0];
	for (size_t i = 1; i < r_host.size(); ++i)
		result += r_host[i];
	return result;
}

template <class F, class T, class... Ts>
auto device_sum_apply(F f, std::span<const T> a, std::span<const Ts>... as)
    -> decltype(f(a[0], as[0]...))
{
	size_t n = a.size();
	assert((as.size() == n && ...));
	using R = decltype(f(a[0], as[0]...));
	constexpr size_t block_size = 256;
	size_t nblocks = (n + block_size - 1) / block_size;
	auto r = device_allocate<R>(nblocks);

	device_sum_apply_kernel<<<nblocks, block_size, block_size * sizeof(R)>>>(
	    f, n, r.data(), a.data(), as.data()...);
	device_synchronize(); // is this necessary?
	auto r_host = device_copy_to_host<R>(r.get());
	auto result = r_host[0];
	for (size_t i = 1; i < r_host.size(); ++i)
		result += r_host[i];
	return result;
}

#endif

}; // namespace mesh