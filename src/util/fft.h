#ifndef UTIL_FFT_H
#define UTIL_FFT_H

/** Some useful wrappers around the FFTW library */

#include <complex>

#include "xtensor/xarray.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xreducer.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
#include <fftw3.h>

#include "util/span.h"

#include <iostream>

/** periodic 2pt-correlator measurement */
class Correlator
{
	size_t size;
	int rep;
	xt::xtensor<std::complex<double>, 2, xt::layout_type::row_major> tmp;
	xt::xarray<double, xt::layout_type::row_major> out;
	fftw_plan plan1, plan2;

  public:
	template <typename G>
	explicit Correlator(G *in, const std::vector<int> &shape)
	{
		rep = G::repSize();
		static_assert(sizeof(G) == G::repSize() * sizeof(double));

		size = 1;
		for (auto s : shape)
		{
			assert(s % 2 == 0);
			size *= s;
		}
		tmp = xt::zeros<std::complex<double>>(
		    {size / shape.back() * (shape.back() / 2 + 1), (size_t)rep});
		out = xt::zeros<double>({(size_t)size});

		int rank = shape.size();

		plan1 = fftw_plan_many_dft_r2c(
		    shape.size(), shape.data(), rep, (double *)in, nullptr, rep, 1,
		    (fftw_complex *)tmp.raw_data(), nullptr, rep, 1, FFTW_MEASURE);
		plan2 = fftw_plan_many_dft_c2r(
		    shape.size(), shape.data(), 1, (fftw_complex *)tmp.raw_data(),
		    nullptr, rep, 1, out.raw_data(), nullptr, 1, 1, FFTW_MEASURE);
	}

	~Correlator()
	{
		fftw_destroy_plan(plan2);
		fftw_destroy_plan(plan1);
	}

	void compute()
	{
		fftw_execute(plan1);
		xt::view(tmp, xt::all(), 0) =
		    xt::sum(tmp * xt::conj(tmp), {1}) / ((double)size * size);
		fftw_execute(plan2);
	}

	xt::xarray<double> &operator()() { return out; }
};

#endif
