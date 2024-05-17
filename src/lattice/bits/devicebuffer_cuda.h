#pragma once

#include "fmt/format.h"
#include "lattice/device.h"
#include "lattice/grid.h"
#include "util/complex.h"
#include "util/hdf5.h"
#include "util/ndarray.h"
#include "util/span.h"
#include "util/stopwatch.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <cuda.h>
#include <type_traits>

namespace mesh {
namespace cuda {

template <class T> class DeviceBuffer;
template <class F, class T, class... Ts>
void device_apply(F f, DeviceBuffer<T> &a, DeviceBuffer<Ts> const &...as);

// memory allocated on device, i.e. a GPU.
template <class T> class DeviceBuffer
{
	// sanity checks on the type T, as we are using (cudas)memcpy liberally and
	// dont bother with constructors/destructors
	static_assert(std::is_trivially_constructible_v<T>);
	static_assert(std::is_trivially_copy_constructible_v<T>);
	static_assert(std::is_trivially_destructible_v<T>);

	unique_device_ptr<T> data_ = {};
	size_t size_ = 0;

  public:
	DeviceBuffer() = default;

	// allocates memory on device
	explicit DeviceBuffer(size_t size)
	    : data_(device_allocate<T>(size)), size_(size)
	{}

	// no implicit copy. use explciit copy instead
	DeviceBuffer(DeviceBuffer const &) = delete;
	DeviceBuffer &operator=(DeviceBuffer const &) = delete;

	// move semantics
	DeviceBuffer(DeviceBuffer &&other) noexcept
	    : data_(std::exchange(other.data_, nullptr)),
	      size_(std::exchange(other.size_, 0))
	{}
	DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
	{
		data_ = std::exchange(other.data_, nullptr);
		size_ = std::exchange(other.size_, 0);
		return *this;
	}

	// device pointers. dont dereference them on the host.
	T *data() { return data_.get(); }
	T const *data() const { return data_.get(); }

	// size metrics
	size_t size() const { return size_; }
	size_t bytes() const { return size_ * sizeof(T); }
	explicit operator bool() const { return data_ != nullptr; }

	void fill_zeros() { device_memclear(data(), size()); }

	void fill(T const &value)
	{
		if (size() == 0)
			return;
		device_apply([value] UTIL_DEVICE(T & a) { a = value; }, *this);
	}

	// create a copy of the data on the devices
	DeviceBuffer copy() const
	{
		auto r = DeviceBuffer(size());
		copy_to(r);
		return r;
	}

	// copy content to another DeviceBuffer (which must have the same size)
	void copy_to(DeviceBuffer &other) const
	{
		if (size() != other.size())
			throw std::runtime_error(
			    "DeviceBuffer: copy_to failed (size mismatch)");
		copy_to(other, 0, 0, size());
	}

	// copy part of this to part of another DeviceBuffer
	void copy_to(DeviceBuffer &dst, size_t dst_offset, size_t src_offset,
	             size_t count) const
	{
		if (dst_offset + count > dst.size())
			throw std::runtime_error(
			    "DeviceBuffer: copy_to failed (dst size mismatch)");
		if (src_offset + count > size())
			throw std::runtime_error(
			    "DeviceBuffer: copy_to failed (src size mismatch)");
		device_copy(dst.data() + dst_offset, data() + src_offset, count);
	}

	void copy_to_2d(DeviceBuffer &dst, size_t dst_offset, size_t dst_pitch,
	                size_t src_offset, size_t src_pitch, size_t width,
	                size_t height) const
	{
		device_copy_2d(dst.data() + dst_offset, dst_pitch, data() + src_offset,
		               src_pitch, width, height);
	}

	void cshift_to(DeviceBuffer &dst, int64_t offset, size_t row_size) const
	{
		// sign convention:
		// *dst = *(src + offset)
		if (&dst == this)
			throw std::runtime_error(
			    "DeviceBuffer: cshift_to failed (dst == src)");
		if (size() != dst.size())
			throw std::runtime_error(
			    "DeviceBuffer: cshift_to failed (size mismatch)");
		if (size() % row_size != 0)
			throw std::runtime_error(
			    "DeviceBuffer: cshift_to failed (invalid row_size)");
		if (offset == 0)
		{
			copy_to(dst);
			return;
		}
		auto height = size() / row_size;

		if (offset > 0)
		{
			copy_to_2d(dst, 0, row_size, offset, row_size, row_size - offset,
			           height);
			copy_to_2d(dst, row_size - offset, row_size, 0, row_size, offset,
			           height);
		}
		else
		{
			copy_to_2d(dst, -offset, row_size, 0, row_size, row_size + offset,
			           height);
			copy_to_2d(dst, 0, row_size, row_size + offset, row_size, -offset,
			           height);
		}
	}

	void copy_from_host(T const *host_data)
	{
		device_copy_from_host(data(), host_data, size());
	}

	void copy_from_host(std::span<const T> host_data)
	{
		if (host_data.size() != size())
		{
			fmt::print("host_data.size() = {}\n", host_data.size());
			fmt::print("size() = {}\n", size());
			assert(false);
		}
		copy_from_host(host_data.data());
	}

	void copy_to_host(T *host_data) const
	{
		device_copy_to_host(host_data, data(), size());
	}

	std::vector<T> copy_to_host() const
	{
		std::vector<T> host_data(size());
		copy_to_host(host_data.data());
		return host_data;
	}
};

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
__global__ void device_sum_kernel(T *r, size_t size, T const *a)
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
__global__ void device_sum_kernel(F f, R *r, size_t size, Ts *...as)
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
void device_apply(F f, DeviceBuffer<T> &a, DeviceBuffer<Ts> const &...as)
{
	assert(((a.size() == as.size()) && ...));
	constexpr size_t block_size = 256; // tunable in [32,1024]
	size_t nblocks = (a.size() + block_size - 1) / block_size;
	device_apply_kernel<<<nblocks, block_size>>>(f, a.size(), a.data(),
	                                             as.data()...);
	device_synchronize();
}

template <class T> auto device_sum(DeviceBuffer<T> const &a) -> T
{
	constexpr size_t block_size = 256;
	size_t nblocks = (a.size() + block_size - 1) / block_size;
	auto r = DeviceBuffer<T>(nblocks);

	device_sum_kernel<<<nblocks, block_size, block_size * sizeof(T)>>>(
	    r.data(), a.size(), a.data());
	device_synchronize(); // is this necessary?
	auto r_host = r.copy_to_host();
	auto result = r_host[0];
	for (size_t i = 1; i < r_host.size(); ++i)
		result += r_host[i];
	return result;
}

template <class F, class T, class... Ts>
auto device_sum(F f, DeviceBuffer<T> const &a, DeviceBuffer<Ts> const &...as)
    -> decltype(f(a.data()[0], as.data()[0]...))
{
	assert(((a.size() == as.size()) && ...));
	using R = decltype(f(a.data()[0], as.data()[0]...));
	constexpr size_t block_size = 256;
	size_t nblocks = (a.size() + block_size - 1) / block_size;
	auto r = DeviceBuffer<R>(nblocks);

	device_sum_kernel<<<nblocks, block_size, block_size * sizeof(R)>>>(
	    f, r.data(), a.size(), a.data(), as.data()...);
	device_synchronize(); // is this necessary?
	auto r_host = r.copy_to_host();
	auto result = r_host[0];
	for (size_t i = 1; i < r_host.size(); ++i)
		result += r_host[i];
	return result;
}

} // namespace cuda

} // namespace mesh
