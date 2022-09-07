#include "lattice/fft.h"

#include <cassert>

namespace mesh {

static constexpr int default_patience = FFTW_MEASURE;

unique_fftw_plan plan_fft(int n, util::complex<double> *in,
                          util::complex<double> *out, int sign, int flags)
{
	auto plan = fftw_plan_dft_1d(n, (fftw_complex *)in, (fftw_complex *)out,
	                             sign, flags);
	if (!plan)
		throw std::runtime_error("fftw_plan_dft_1d() failed");
	return unique_fftw_plan(plan);
}

unique_fftw_plan plan_fft_all(std::span<const int> const shape,
                              util::complex<double> *in,
                              util::complex<double> *out, int sign, int flags)
{
	auto plan = fftw_plan_dft(shape.size(), shape.data(), (fftw_complex *)in,
	                          (fftw_complex *)out, sign, flags);
	if (!plan)
		throw std::runtime_error("fftw_plan_dft() failed");
	return unique_fftw_plan(plan);
}

unique_fftw_plan plan_fft_last(std::span<const int> shape,
                               util::complex<double> *in,
                               util::complex<double> *out, int sign, int flags)
{
	assert(shape.size() >= 1);
	int howmany = 1; // number of independent FFTs
	for (size_t i = 0; i < shape.size() - 1; ++i)
		howmany *= shape[i];
	int dist = shape.back(); // distance from one FFT to the next
	int stride = 1;          // stride inside one FFT
	auto plan = fftw_plan_many_dft(
	    1, &shape.back(), howmany, (fftw_complex *)in, nullptr, stride, dist,
	    (fftw_complex *)out, nullptr, stride, dist, sign, flags);
	if (!plan)
		throw std::runtime_error("fftw_plan_many_dft() failed");
	return unique_fftw_plan(plan);
}

Lattice<util::complex<double>> fft_all(Lattice<util::complex<double>> &in,
                                       int sign)
{
	auto &grid = in.grid();
	assert(grid.isize() == 1);
	auto out = Lattice<util::complex<double>>(grid);

	// TODO: Use flags = FFTW_MEASURE | FFTW_WISDOM_ONLY and run measurment in
	//       separate allocation when needed.
	auto plan =
	    plan_fft_all(grid.shape(), in.data(), out.data(), sign, FFTW_ESTIMATE);
	fftw_execute(plan.get());
	return out;
}

Lattice<util::complex<double>> fft_last(Lattice<util::complex<double>> &in,
                                        int sign)
{
	auto &grid = in.grid();
	assert(grid.isize() == 1);
	auto out = Lattice<util::complex<double>>(grid);

	// TODO: Use flags = FFTW_MEASURE | FFTW_WISDOM_ONLY and run measurment in
	//       separate allocation when needed.
	auto plan =
	    plan_fft_last(grid.shape(), in.data(), out.data(), sign, FFTW_ESTIMATE);
	fftw_execute(plan.get());
	return out;
}

std::vector<Coordinate> make_mom_list(int nd, int mom2max)
{
	std::vector<Coordinate> r;
	int m = int(std::sqrt(mom2max));

	if (nd == 1)
	{
		for (int x = -m; x <= m; ++x)
			if (x * x <= mom2max)
				r.push_back({x});
	}
	else if (nd == 2)
	{
		for (int x = -m; x <= m; ++x)
			for (int y = -m; y <= m; ++y)
				if (x * x + y * y <= mom2max)
					r.push_back({x, y});
	}
	else if (nd == 3)
	{
		for (int x = -m; x <= m; ++x)
			for (int y = -m; y <= m; ++y)
				for (int z = -m; z <= m; ++z)
					if (x * x + y * y + z * z <= mom2max)
						r.push_back({x, y, z});
	}
	else if (nd == 4)
	{
		for (int x = -m; x <= m; ++x)
			for (int y = -m; y <= m; ++y)
				for (int z = -m; z <= m; ++z)
					for (int t = -m; t <= m; ++t)
						if (x * x + y * y + z * z + t * t <= mom2max)
							r.push_back({x, y, z, t});
	}
	else
		assert(false);
	return r;
}

} // namespace mesh