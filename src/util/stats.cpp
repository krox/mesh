#include "util/stats.h"

#include <algorithm>
#include <cmath>
#include <fmt/format.h>

ConstantFit::ConstantFit(const std::vector<double> &ys)
{
	a = 0;
	for (double y : ys)
		a += y;
	a /= ys.size();
	a_err = 0.0 / 0.0;
}

ConstantFit::ConstantFit(const std::vector<double> &ys,
                         const std::vector<double> &ys_err)
{
	assert(ys.size() == ys_err.size());
	a = 0;
	a_err = 0;
	for (size_t i = 0; i < ys.size(); ++i)
	{
		a += ys[i] / (ys_err[i] * ys_err[i]);
		a_err += 1 / (ys_err[i] * ys_err[i]);
	}
	a /= a_err;
	a_err = 1 / sqrt(a_err);
}

double ConstantFit::operator()() const { return a; }
double ConstantFit::operator()([[maybe_unused]] double x) const { return a; }

LinearFit::LinearFit(const std::vector<double> &xs,
                     const std::vector<double> &ys)
{
	assert(xs.size() == ys.size());
	Estimator<2> est;
	for (size_t i = 0; i < xs.size(); ++i)
		est.add({xs[i], ys[i]});
	b = est.cov(0, 1) / est.var(0);
	a = est.mean(1) - est.mean(0) * b;
}

LinearFit::LinearFit(const std::vector<double> &xs,
                     const std::vector<double> &ys,
                     const std::vector<double> &ys_err)
{
	assert(xs.size() == ys.size());
	Estimator<2> est;
	for (size_t i = 0; i < xs.size(); ++i)
		est.add({xs[i], ys[i]}, 1.0 / (ys_err[i] * ys_err[i]));
	b = est.cov(0, 1) / est.var(0);
	a = est.mean(1) - est.mean(0) * b;
}

/** evaluate the fitted function */
double LinearFit::operator()(double x) const { return a + b * x; }

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
	return add(x, 1.0);
}

template <size_t dim>
void Estimator<dim>::add(std::array<double, dim> x, double w)
{
	n += w;
	double dx[dim];
	for (size_t i = 0; i < dim; ++i)
	{
		dx[i] = x[i] - avg[i];
		avg[i] += dx[i] * (w / n);
	}

	for (size_t i = 0; i < dim; ++i)
		for (size_t j = 0; j < dim; ++j)
			sum2[i][j] += w * dx[i] * (x[j] - avg[j]);
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

double Autocorrelation::corrTime() const
{
	// IDEA: use data log(abs(corr(i))) and do a linear fit
	double sumXY = 0, sumXX = 0, sumW = 0;
	for (size_t i = 0; i < len && ac[i].n >= 2; ++i)
	{
		double c = fabs(corr(i));
		double w = 1.0 / (c * c);
		sumW += w;
		sumXX += w * i * i;
		sumXY += w * i * log(c);
	}
	return -sumXX / sumXY;
}

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
