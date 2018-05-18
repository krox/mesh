#ifndef UTIL_NUMERICS_H
#define UTIL_NUMERICS_H

/**
 * Basic numeric helpers.
 * Root finding and integration in one dimension.
 */

#include <functional>
#include <stdexcept>

/** function type used throughout this module */
typedef std::function<double(double)> function_t;

/** exception thrown if some method does not converge */
class numerics_exception : std::runtime_error
{
  public:
	numerics_exception(const std::string &what_arg)
	    : std::runtime_error(what_arg)
	{}

	numerics_exception(const char *what_arg) : std::runtime_error(what_arg) {}
};

/**
 * Solve f(x) = 0 for x in [a,b].
 * Implemented using Secant method with fallback to bisection.
 */
double solve(function_t f, double a, double b);

/**
 * Integrate f(x) for x in [a, b].
 * Implemented using adaptive Gauss-Kronrod quadrature.
 */
double integrate(function_t f, double a, double b);
double integrate(function_t f, double a, double b, double eps, int maxCalls);

#endif
