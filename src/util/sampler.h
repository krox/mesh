#ifndef UTIL_SAMPLER_H
#define UTIL_SAMPLER_H

#include <cassert>
#include <cmath>
#include <queue>
#include <random>
#include <vector>

#include "util/numerics.h"

/**
 * Sampler for arbitrary distribution
 */
class LogSampler
{
	/** log-density function and its derivatives */
	function_t f, fd, fdd;

	double min, max;

	struct Region
	{
		double a, b; // left/right bounds of region

		// upper/lower bound of f in the interval [a,b]
		// alpha + beta*x >= f(a+x(b-a)) >= gamma + delta*x
		double alpha, beta;
		double gamma, delta;

		/** area of upper/lower bound */
		double areaUpper() const;
		double areaLower() const;

		/** lower area / upper area */
		double frac() const;

		/** sort regions with bad bounds to the front */
		bool operator<(const Region &r) const;
	};

	std::vector<Region> regs;

	std::discrete_distribution<int> disc_dist;
	std::uniform_real_distribution<double> uni_dist;

	size_t nTries = 0;
	size_t nEvals = 0;
	size_t nAccepts = 0;

	Region makeRegion(double a, double b);

  public:
	/** constructor */
	LogSampler(function_t f, function_t fd, function_t fdd, double min,
	           double max, size_t nRegs = 50);

	/** generate one sample */
	template <typename Rng> double operator()(Rng &rng);

	/** should be close to 1 (if not, increase nRegs) */
	double quality() const;

	/** accaptance/evaluation-probability so far */
	double accProb() const { return (double)nAccepts / nTries; }
	double evalProb() const { return (double)nEvals / nTries; }

	/** generate some samples and plot a histogram */
	void test();
};

template <typename Rng> inline double LogSampler::operator()(Rng &rng)
{
	while (true)
	{
		++nTries;

		// choose a region
		size_t i = disc_dist(rng);
		const Region &r = regs[i];

		// candidate in [0,1] with exponential distribution
		double u = uni_dist(rng) * (1 - exp(-r.beta)) + exp(-r.beta);
		double x = log(u) / r.beta + 1;
		if (r.beta == 0)
			x = uni_dist(rng);
		assert(0 <= x && x <= 1);

		// uniform in [0,upper(x)]
		double y = uni_dist(rng) * exp(r.alpha + r.beta * x);

		if (y > exp(r.gamma + r.delta * x)) // over lower bound?
		{
			++nEvals;
			if (y > exp(f(r.a + (r.b - r.a) * x))) // over f itself?
				continue;
		}

		++nAccepts;
		return regs[i].a + (r.b - r.a) * x;
	}
}

#endif
