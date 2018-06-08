#ifndef UTIL_SPAN_H
#define UTIL_SPAN_H

#include <vector>

template <typename T> class span
{
	T *data_ = nullptr;
	size_t size_ = 0;

  public:
	/** constructors */
	span() = default;
	span(T *data, size_t size) : data_(data), size_(size) {}
	span(T *begin, T *end) : data_(begin), size_(end - begin) {}
	span(std::vector<T> &v) : data_(v.data()), size_(v.size()) {}

	/** field access */
	T *data() { return data_; }
	const T *data() const { return data_; }
	size_t size() const { return size_; }

	/** element access */
	T &operator[](size_t i) { return data_[i]; }
	const T &operator[](size_t i) const { return data_[i]; }

	/** iterators */
	T *begin() { return data_; }
	T *end() { return data_ + size_; }
	const T *begin() const { return data_; }
	const T *end() const { return data_ + size_; }
};

#endif
