#include "lattice/device.h"
#include "fmt/format.h"
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdexcept>

namespace mesh {

namespace {
void check(cudaError_t err)
{
	if (err == cudaSuccess)
		return;

	auto msg = cudaGetErrorString(err);
	throw std::runtime_error(fmt::format("cuda error: {}", msg));
}
} // namespace

void device_synchronize() { check(cudaDeviceSynchronize()); }

void *device_malloc(size_t n)
{
	if (n == 0)
		return nullptr;
	void *ptr;
	check(cudaMalloc(&ptr, n));
	assert(ptr != nullptr);
	return ptr;
}

void device_free(void *ptr)
{
	if (ptr == nullptr)
		return;
	check(cudaFree(ptr));
}

void device_memcpy(void *dst, void const *src, size_t n)
{
	if (n == 0)
		return;
	check(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice));
}

void device_memcpy_2d(void *dst, size_t dpitch, void const *src, size_t spitch,
                      size_t width, size_t height)
{
	if (width == 0 || height == 0)
		return;
	check(cudaMemcpy2D(dst, dpitch, src, spitch, width, height,
	                   cudaMemcpyDeviceToDevice));
}

void device_memcpy_to_host(void *dst, void const *src, size_t n)
{
	if (n == 0)
		return;
	check(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost));
}

void device_memcpy_from_host(void *dst, void const *src, size_t n)
{
	if (n == 0)
		return;
	check(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
}

void device_memclear(void *ptr, size_t n)
{
	if (n == 0)
		return;
	check(cudaMemset(ptr, 0, n));
}

} // namespace mesh