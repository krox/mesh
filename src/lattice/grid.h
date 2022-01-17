#pragma once

#include "util/span.h"
#include "util/static_vector.h"
#include <map>

namespace mesh {

using Coordinate = util::static_vector<int32_t, 4>;

class Grid
{
	static std::string makeDescription(Coordinate const &shape,
	                                   Coordinate const &ishape)
	{
		auto sshape = util::span<const int32_t>(shape.begin(), shape.end());
		auto sishape = util::span<const int32_t>(ishape.begin(), ishape.end());
		return fmt::format("{{{}, {}}}", sshape, sishape);
	}

	size_t osize_, isize_;
	Coordinate oshape_, ishape_;
	std::string desc_; // human readable description

	// this is a trick to make the constructor "effectively private" and still
	// enabling delegated construction as in '.emplace' or similar functions
	struct private_key
	{};

  public:
	Grid(Coordinate const &shape, Coordinate const &ishape, private_key)
	    : osize_(1), isize_(1), oshape_(shape), ishape_(ishape),
	      desc_(makeDescription(shape, ishape))
	{
		assert(oshape_.size() == ishape_.size());
		for (size_t i = 0; i < oshape_.size(); ++i)
		{
			assert(oshape_[i] % ishape_[i] == 0);
			oshape_[i] /= ishape_[i];
			osize_ *= oshape_[i];
			isize_ *= ishape_[i];
		}
	}

	/** disable copy/move. Usercode should only use pointers created by make */
	Grid(Grid const &) = delete;
	Grid &operator=(Grid const &) = delete;

	/** geometry of the lattice */
	int ndim() const { return (int)oshape_.size(); }
	size_t osize() const { return osize_; }
	size_t isize() const { return isize_; }
	size_t size() const { return osize_ * isize_; }
	int32_t oshape(size_t i) const { return oshape_[i]; }
	int32_t ishape(size_t i) const { return ishape_[i]; }
	int32_t shape(size_t i) const { return oshape_[i] * ishape_[i]; }

	/** convert coordinate into (outerInde, innerIndex) */
	std::pair<size_t, size_t> flatIndex(Coordinate const &index) const
	{
		assert((int)index.size() == ndim());
		size_t oIndex = 0;
		size_t iIndex = 0;
		for (size_t i = 0; i < index.size(); ++i)
		{
			assert(index[i] < shape(i));
			oIndex = oIndex * oshape(i) + index[i] % oshape(i);
			iIndex = iIndex * ishape(i) + index[i] / oshape(i);
		}
		assert(oIndex < osize());
		assert(iIndex < isize());
		return {oIndex, iIndex};
	}

	/** human-readable description */
	std::string const &to_string() const { return desc_; }

	/** "constructors" */
	static Grid const &make(Coordinate const &shape, Coordinate const &ishape)
	{
		// NOTE: std::map guarantees pointer-stability
		static std::map<std::string, Grid> cache;
		assert(shape.size() == ishape.size());

		auto [it, _] = cache.try_emplace(makeDescription(shape, ishape), shape,
		                                 ishape, private_key{});
		return it->second;
	}

	/** automatic simd layout */
	static Grid const &make(Coordinate const &shape, int simdWidth)
	{
		auto ishape = Coordinate(shape.size(), 1);
		assert(simdWidth != 0 && (simdWidth & (simdWidth - 1)) == 0);
		while (simdWidth > 1)
		{
			int best = -1;
			int bestSize = 0;
			for (int i = 0; i < (int)shape.size(); ++i)
				if ((shape[i] / ishape[i]) % 2 == 0)
					if (shape[i] / ishape[i] >= bestSize)
					{
						bestSize = shape[i] / ishape[i];
						best = i;
					}
			if (best == -1)
				throw std::runtime_error("unable to distribute all simd lanes "
				                         "over lattice geometry");
			ishape[best] *= 2;
			simdWidth /= 2;
		}
		return make(shape, ishape);
	}
};

} // namespace mesh
