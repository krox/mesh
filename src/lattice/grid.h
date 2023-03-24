#pragma once

#include "fmt/ranges.h"
#include "lattice/tensor.h"
#include "util/vector.h"
#include <map>
#include <span>

namespace mesh {

using Coordinate = util::static_vector<int32_t, 4>;

class Grid
{
	size_t size_ = 0;
	Coordinate shape_ = {0};

	// FUTURE: description of MPI layout should go here too

  public:
	// NOTE: the default constructor creates a 1-dimensional grid with 0
	//       elements. A 0-dimensional grid would need 1 element, which would
	//       result in a (pointless) memory allocation when creating a lattice.
	Grid() = default;

	explicit Grid(Coordinate const &shape) : size_(1), shape_(shape)
	{
		for (size_t i = 0; i < shape_.size(); ++i)
			size_ *= shape_[i];
	}

	// geometry of the lattice
	int ndim() const { return (int)shape_.size(); }
	size_t size() const { return size_; }
	Coordinate shape() const { return shape_; }
	int32_t shape(size_t i) const { return shape_[i]; }

	// convert coordinate into single flat index
	size_t flat_index(Coordinate const &index) const
	{
		assert((int)index.size() == ndim());
		size_t r = 0;
		for (int i = 0; i < ndim(); ++i)
		{
			assert(0 <= index[i] && index[i] < shape(i));
			r = r * shape(i) + index[i];
		}
		return r;
	}

	// human-readable description
	std::string to_string() const
	{
		return fmt::format("{}", std::span(shape_.begin(), shape_.end()));
	}

	bool operator==(Grid const &) const = default;
};

} // namespace mesh
