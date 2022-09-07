#pragma once

// Some useful wrappers around the FFTW3 library.
// Also some helpers for "slow Fourier transforms" (contractions with e^-ipx for
// one or few momenta)

#include "lattice/lattice.h"
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

// thin wrappers around fftw_plan_*, mostly to adapt types a little bit
//     * throws on errors
//     * FFTW_FORWARD = -1, FFTW_BACKWARD = 1, though this is just convention
//     * by default, planning destoys both input and output
//     * for optimal performance, arrays should be aligned, for example using
//           util::aligned_allocate<util::complex<double>>(...)

// 1D transform
unique_fftw_plan plan_fft(int n, util::complex<double> *in,
                          util::complex<double> *out, int sign,
                          int flags = FFTW_MEASURE);

// transforms all axes
unique_fftw_plan plan_fft_all(std::span<const int> shape,
                              util::complex<double> *in,
                              util::complex<double> *out, int sign,
                              int flags = FFTW_MEASURE);

// transforms just the last axis
unique_fftw_plan plan_fft_last(std::span<const int> shape,
                               util::complex<double> *in,
                               util::complex<double> *out, int sign,
                               int flags = FFTW_MEASURE);

// 'Lattice' based interface
//     - only for non-vectorized lattices, though FFTW does its own
//       vectorization internally regardless.
Lattice<util::complex<double>> fft_all(Lattice<util::complex<double>> &,
                                       int sign);
Lattice<util::complex<double>> fft_last(Lattice<util::complex<double>> &,
                                        int sign);

// list momenta with |p|^2 < mom2max
std::vector<Coordinate> make_mom_list(int nd, int mom2max);

// make_phases(std::span<const int> shape)

} // namespace mesh
