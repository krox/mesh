#include "lattice/fft.h"

#include <cassert>

namespace mesh {

namespace {
unique_fftw_plan make(fftw_plan p)
{
	if (!p)
		throw std::runtime_error("could not create FFTW plan");
	return unique_fftw_plan(p);
}
} // namespace

unique_fftw_plan plan_fft(int n, util::complex<double> *in,
                          util::complex<double> *out)
{
	return make(fftw_plan_dft_1d(n, (fftw_complex *)in, (fftw_complex *)out,
	                             FFTW_FORWARD, FFTW_ESTIMATE));
}

unique_fftw_plan plan_ifft(int n, util::complex<double> *in,
                           util::complex<double> *out)
{
	return make(fftw_plan_dft_1d(n, (fftw_complex *)in, (fftw_complex *)out,
	                             FFTW_BACKWARD, FFTW_ESTIMATE));
}

unique_fftw_plan plan_fft_all(std::span<const int> const shape,
                              util::complex<double> *in,
                              util::complex<double> *out)
{
	return make(fftw_plan_dft(shape.size(), shape.data(), (fftw_complex *)in,
	                          (fftw_complex *)out, FFTW_FORWARD,
	                          FFTW_ESTIMATE));
}

unique_fftw_plan plan_ifft_all(std::span<const int> const shape,
                               util::complex<double> *in,
                               util::complex<double> *out)
{
	return make(fftw_plan_dft(shape.size(), shape.data(), (fftw_complex *)in,
	                          (fftw_complex *)out, FFTW_BACKWARD,
	                          FFTW_ESTIMATE));
}

namespace {
unique_fftw_plan plan_fft_last_impl(std::span<const int> shape,
                                    util::complex<double> *in,
                                    util::complex<double> *out, int sign)
{
	assert(shape.size() >= 1);
	int howmany = 1; // number of independent FFTs
	for (size_t i = 0; i < shape.size() - 1; ++i)
		howmany *= shape[i];
	int dist = shape.back(); // distance from one FFT to the next
	int stride = 1;          // stride inside one FFT
	return make(fftw_plan_many_dft(
	    1, &shape.back(), howmany, (fftw_complex *)in, nullptr, stride, dist,
	    (fftw_complex *)out, nullptr, stride, dist, sign, FFTW_ESTIMATE));
}
} // namespace

unique_fftw_plan plan_fft_last(std::span<const int> shape,
                               util::complex<double> *in,
                               util::complex<double> *out)
{
	return plan_fft_last_impl(shape, in, out, FFTW_FORWARD);
}

unique_fftw_plan plan_ifft_last(std::span<const int> shape,
                                util::complex<double> *in,
                                util::complex<double> *out)
{
	return plan_fft_last_impl(shape, in, out, FFTW_BACKWARD);
}

} // namespace mesh