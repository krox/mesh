#pragma once

// Some useful wrappers around the FFTW3 library

#include "util/complex.h"
#include "util/memory.h"
#include <algorithm>
#include <fftw3.h>
#include <numeric>
#include <span>
#include <vector>

namespace mesh {

// 'fftw_plan' is indeed a pointer (though the pointed-to type is unnamed),
// so we can use std::unique_ptr for a nice RAII wrapper
static_assert(std::is_pointer_v<fftw_plan>);
struct fftw_plan_delete
{
	void operator()(fftw_plan p) { fftw_destroy_plan(p); }
};
using unique_fftw_plan =
    std::unique_ptr<std::remove_pointer_t<fftw_plan>, fftw_plan_delete>;

// thin wrappers around fftw_plan_*
unique_fftw_plan plan_fft(int n, util::complex<double> *in,
                          util::complex<double> *out);
unique_fftw_plan plan_ifft(int n, util::complex<double> *in,
                           util::complex<double> *out);

// forward/backward FFT of all axes
unique_fftw_plan plan_fft_all(std::span<const int> shape,
                              util::complex<double> *in,
                              util::complex<double> *out);
unique_fftw_plan plan_ifft_all(std::span<const int> shape,
                               util::complex<double> *in,
                               util::complex<double> *out);

// forward/backward FFT of the last axis
unique_fftw_plan plan_fft_last(std::span<const int> shape,
                               util::complex<double> *in,
                               util::complex<double> *out);
unique_fftw_plan plan_ifft_last(std::span<const int> shape,
                                util::complex<double> *in,
                                util::complex<double> *out);

// measures 2-pt correlation function in spatial momentum space
//     * owns both input and output memory (more natural for fftw)
//     * can(/should) be used multiple times. Just overwrite the 'in'
//       array, and call 'run()' again
class Correlator
{
	size_t size_;
	util::memory_ptr<util::complex<double>> in_, tmp_, out_;
	unique_fftw_plan plan1_, plan2_;

  public:
	// not needed (yet?)
	Correlator(Correlator const &) = delete;
	Correlator &operator=(Correlator const &) = delete;

	explicit Correlator(std::span<const int> shape)
	    : size_(std::reduce(shape.begin(), shape.end(), 1, std::multiplies{})),
	      in_(util::aligned_allocate<util::complex<double>>(size_)),
	      tmp_(util::aligned_allocate<util::complex<double>>(size_)),
	      out_(util::aligned_allocate<util::complex<double>>(size_)),
	      plan1_(plan_fft_all(shape, in_.get(), tmp_.get())),
	      plan2_(plan_ifft_last(shape, tmp_.get(), out_.get()))
	{}

	std::span<util::complex<double>> in()
	{
		return std::span(in_.get(), size_);
	}
	std::span<util::complex<double>> out()
	{
		return std::span(out_.get(), size_);
	}

	void run()
	{
		// NOTE: the 'c2r' ('plan2') destroys its input ('tmp_').
		fftw_execute(plan1_.get());
		for (size_t i = 0; i < size_; ++i)
			tmp_[i] = norm2(tmp_[i]);
		fftw_execute(plan2_.get());
	}
};

} // namespace mesh
