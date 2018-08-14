#ifndef UTIL_FFT_H
#define UTIL_FFT_H

/** Some useful wrappers around the FFTW library */

#include <complex>

#include <fftw3.h>

#include "util/span.h"

#include <iostream>

/** periodic 1D correlation function */
class Correlator
{
	int n;
	double *in, *out;
	fftw_complex *tmp;
	fftw_plan plan1, plan2;

  public:
	explicit Correlator(int n)
	    : n(n), in(fftw_alloc_real(n)), out(fftw_alloc_real(n)),
	      tmp(fftw_alloc_complex(n / 2 + 1))
	{
		assert(n > 0);
		plan1 = fftw_plan_dft_r2c_1d(n, in, tmp, FFTW_MEASURE);
		plan2 = fftw_plan_dft_c2r_1d(n, tmp, out, FFTW_MEASURE);
	}

	~Correlator()
	{
		fftw_destroy_plan(plan2);
		fftw_destroy_plan(plan1);
		fftw_free(in);
		fftw_free(out);
		fftw_free(tmp);
	}

	void compute(span<const double> x)
	{
		assert(x.size() == (size_t)n);
		for (int i = 0; i < n; ++i)
			in[i] = x[i];
		fftw_execute(plan1);
		for (int i = 0; i < n / 2 + 1; ++i)
		{
			tmp[i][0] =
			    (tmp[i][0] * tmp[i][0] + tmp[i][1] * tmp[i][1]) * (1.0 / n / n);
			tmp[i][1] = 0;
		}
		fftw_execute(plan2);
	}

	span<const double> operator()() const { return span<const double>(out, n); }
	operator span<const double>() const { return span<const double>(out, n); }
};

#endif
