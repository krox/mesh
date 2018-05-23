#include "util/stats.h"

#include <algorithm>
#include <cmath>
#include <fmt/format.h>

histogram::histogram(double min, double max, size_t n)
    : mins(n), maxs(n), bins(n, 0)
{
	for (size_t i = 0; i < n; ++i)
	{
		mins[i] = min + (max - min) * i / n;
		maxs[i] = min + (max - min) * (i + 1) / n;
	}
}

void histogram::add(double x)
{
	auto it = std::upper_bound(maxs.begin(), maxs.end(), x);
	auto i = std::distance(maxs.begin(), it);
	if (0 <= i && i < (ptrdiff_t)bins.size()) // ignore out-of-range samples
		bins[i] += 1;
}

template <size_t dim> Estimator<dim>::Estimator() { clear(); }

template <size_t dim> void Estimator<dim>::add(std::array<double, dim> x)
{
	n += 1;
	double dx[dim];
	for (size_t i = 0; i < dim; ++i)
	{
		dx[i] = x[i] - avg[i];
		avg[i] += dx[i] / n;
	}

	for (size_t i = 0; i < dim; ++i)
		for (size_t j = 0; j < dim; ++j)
			sum2[i][j] += dx[i] * (x[j] - avg[j]);
}

template <size_t dim> double Estimator<dim>::mean(size_t i) const
{
	return avg[i];
}

template <size_t dim> double Estimator<dim>::var(size_t i) const
{
	return sum2[i][i] / (n - 1);
}

template <size_t dim> double Estimator<dim>::cov(size_t i, size_t j) const
{
	return sum2[i][j] / (n - 1);
}

template <size_t dim> double Estimator<dim>::corr(size_t i, size_t j) const
{
	return cov(i, j) / sqrt(var(i) * var(j));
}

template <size_t dim> void Estimator<dim>::clear()
{
	n = 0;
	for (size_t i = 0; i < dim; ++i)
		avg[i] = 0;
	for (size_t i = 0; i < dim; ++i)
		for (size_t j = 0; j < dim; ++j)
			sum2[i][j] = 0;
}

template class Estimator<1>;
template class Estimator<2>;
template class Estimator<3>;
template class Estimator<4>;

void Autocorrelation::add(double x)
{
	history[count % len] = x;
	for (size_t i = 0; i < std::min(count + 1, len); ++i)
		ac[i].add({x, history[(count - i) % len]});
	count += 1;
}

double Autocorrelation::mean() const { return ac[0].mean(); }
double Autocorrelation::var() const { return ac[0].var(); }

double Autocorrelation::cov(int lag) const { return ac[lag].cov(); }
double Autocorrelation::corr(int lag) const { return ac[lag].corr(); }

void Autocorrelation::write(size_t maxLen) const
{
	for (size_t i = 0; i < maxLen; ++i)
		fmt::print("{:>2}: {:.2f}", i, corr(i));
}

void Autocorrelation::clear()
{
	count = 0;
	for (size_t i = 0; i < len; ++i)
		ac[i].clear();
}
