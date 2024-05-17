#pragma once

// This module handles some low-level management of device memory.
// Note that this header does not need to be compiled with nvcc.

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
struct device_deleter
{
	void operator()(void *ptr) noexcept { device_free(ptr); }
};
template <class T> using unique_device_ptr = std::unique_ptr<T, device_deleter>;

template <class T> unique_device_ptr<T> device_allocate(size_t n)
{
	return unique_device_ptr<T>(static_cast<T *>(device_malloc(n * sizeof(T))));
}

// typed memory movement for convenience
template <class T> void device_copy(T *dst, T const *src, size_t n)
{
	device_memcpy(dst, src, n * sizeof(T));
}

template <class T>
void device_copy_2d(T *dst, size_t dpitch, T const *src, size_t spitch,
                    size_t width, size_t height)
{
	device_memcpy_2d(dst, dpitch * sizeof(T), src, spitch * sizeof(T),
	                 width * sizeof(T), height);
}

// set memory to zero (NOTE: this is bit-wise zero, ignoring the type T)
template <class T> void device_clear(T *ptr, size_t n)
{
	device_memclear(ptr, n * sizeof(T));
}

template <class T> void device_copy_to_host(T *dst, T const *src, size_t n)
{
	device_memcpy_to_host(dst, src, n * sizeof(T));
}

template <class T> void device_copy_from_host(T *dst, T const *src, size_t n)
{
	device_memcpy_from_host(dst, src, n * sizeof(T));
}

}; // namespace mesh