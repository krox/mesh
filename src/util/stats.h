#ifndef MESH_STATS_H
#define MESH_STATS_H

/** statistics utilities */

#include <array>
#include <vector>

struct histogram
{
	std::vector<double> mins;
	std::vector<double> maxs;
	std::vector<size_t> bins;

	histogram(double min, double max, size_t n);
	histogram(const std::vector<double> &xs);

	void add(double x);
};

/**
 * Estimate mean/variance/covariance of a population as samples are coming in.
 * This is the same as the standard formula "Var(x) = n/(n-1) (E(x^2) - E(x)^2)"
 * but numerically more stable.
 */
template <size_t dim> struct Estimator
{
	size_t n = 0;
	double avg[dim];       // = 1/n ∑ x_i
	double sum2[dim][dim]; //= ∑ (x_i - meanX)*(y_i - meanY)

  public:
	/** default constructor */
	Estimator();

	/** add a new data point */
	void add(std::array<double, dim> x);

	/** mean/variance in dimension i */
	double mean(size_t i = 0) const;
	double var(size_t i = 0) const;

	/** covariance/correlation between dimensions i and j */
	double cov(size_t i = 0, size_t j = 1) const;
	double corr(size_t i = 0, size_t j = 1) const;

	/** reset everything */
	void clear();
};

/** analyze autocorrelation of a single stream of data */
class Autocorrelation
{
	static constexpr size_t len = 20;

	size_t count = 0;
	double history[len]; // previously added values
	Estimator<2> ac[len];

  public:
	/** default constructor */
	Autocorrelation() = default;

	/** add a new data point */
	void add(double x);

	/** mean/variance of data points */
	double mean() const;
	double var() const;

	/** covariance/correlation between data[i] and data[i-lag] */
	double cov(int lag = 1) const;
	double corr(int lag = 1) const;

	/** print analysis to stdout */
	void write(size_t maxLen = len) const;

	/** reset everything */
	void clear();
};

#endif
