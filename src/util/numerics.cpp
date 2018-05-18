#include "util/numerics.h"

#include <cassert>
#include <cmath>
#include <queue>
using namespace std;

double solve(function_t f, double a, double b)
{
	assert(!isnan(a) && !isnan(b));

	double fa = f(a);
	double fb = f(b);
	assert(!std::isnan(fa) && !isnan(fb));
	if (fa == 0)
		return a;
	if (fb == 0)
		return b;
	assert(signbit(fa) != signbit(fb));
	double c = b;
	double fc = fb;

	// a should be the best guess
	if (abs(fb) < abs(fa))
	{
		swap(a, b);
		swap(fa, fb);
	}

	for (int iter = 0; iter < 100; ++iter)
	{
		// choose new point x
		double x = (b * fa - a * fb) / (fa - fb); // secant method

		// outside bracket (or nan) -> fall back to bisection
		if (!(a < x && x < c) && !(c < x && x < a))
		{
			x = 0.5 * (a + c);
			if (x == a || x == c) // there is no further floating point number
			                      // between a and b -> we are done
			{
				if (abs(fc) < abs(fa))
					return c;
				else
					return a;
			}
		}

		// evaluate f at new point
		b = a;
		fb = fa;
		a = x;
		fa = f(x);
		assert(!isnan(fa));
		if (fa == 0)
			return a;

		// update brackets
		if (signbit(fa) != signbit(fb))
		{
			c = b;
			fc = fb;
		}
	}

	throw new numerics_exception("secant method did not converge");
}

// Gauss/Kronrod nodes
static const double GK31_x[] = {
    0.000000000000000000000000000000000e+00,
    2.011940939974345223006283033945962e-01,
    3.941513470775633698972073709810455e-01,
    5.709721726085388475372267372539106e-01,
    7.244177313601700474161860546139380e-01,
    8.482065834104272162006483207742169e-01,
    9.372733924007059043077589477102095e-01,
    9.879925180204854284895657185866126e-01,

    1.011420669187174990270742314473923e-01,
    2.991800071531688121667800242663890e-01,
    4.850818636402396806936557402323506e-01,
    6.509967412974169705337358953132747e-01,
    7.904185014424659329676492948179473e-01,
    8.972645323440819008825096564544959e-01,
    9.677390756791391342573479787843372e-01,
    9.980022986933970602851728401522712e-01,
};

// Gauss weights
static const double GK31_wg[] = {
    2.025782419255612728806201999675193e-01,
    1.984314853271115764561183264438393e-01,
    1.861610000155622110268005618664228e-01,
    1.662692058169939335532008604812088e-01,
    1.395706779261543144478047945110283e-01,
    1.071592204671719350118695466858693e-01,
    7.036604748810812470926741645066734e-02,
    3.075324199611726835462839357720442e-02,
};

// Kronrod weights
static const double GK31_wk[] = {
    1.013300070147915490173747927674925e-01,
    9.917359872179195933239317348460313e-02,
    9.312659817082532122548687274734572e-02,
    8.308050282313302103828924728610379e-02,
    6.985412131872825870952007709914748e-02,
    5.348152469092808726534314723943030e-02,
    3.534636079137584622203794847836005e-02,
    1.500794732931612253837476307580727e-02,

    1.007698455238755950449466626175697e-01,
    9.664272698362367850517990762758934e-02,
    8.856444305621177064727544369377430e-02,
    7.684968075772037889443277748265901e-02,
    6.200956780067064028513923096080293e-02,
    4.458975132476487660822729937327969e-02,
    2.546084732671532018687400101965336e-02,
    5.377479872923348987792051430127650e-03,
};

/** returns (Gauss,Kronrod) quadrature using 15/31 function evaluations */
static pair<double, double> integrateKronrod31(function_t f, double a, double b)
{
	double mid = (a + b) / 2;
	double half = (b - a) / 2;

	double f0 = f(mid);
	double sumG = GK31_wg[0] * f0;
	double sumK = GK31_wk[0] * f0;
	for (size_t i = 1; i < 8; ++i)
	{
		f0 = f(mid - half * GK31_x[i]) + f(mid + half * GK31_x[i]);
		sumG += GK31_wg[i] * f0;
		sumK += GK31_wk[i] * f0;
	}
	for (size_t i = 8; i < 16; ++i)
	{
		f0 = f(mid - half * GK31_x[i]) + f(mid + half * GK31_x[i]);
		sumK += GK31_wk[i] * f0;
	}

	sumG *= half;
	sumK *= half;
	return {sumG, sumK};
}

struct Region
{
	double a, b;
	double val, err;

	Region(function_t f, double a, double b) : a(a), b(b)
	{
		auto est = integrateKronrod31(f, a, b);
		val = est.second;
		err = abs(est.first - est.second);
	}

	bool operator<(const Region &b) const { return err < b.err; }
};

double integrate(function_t f, double a, double b, double eps, int maxCalls)
{
	priority_queue<Region> q;

	auto reg = Region(f, a, b);
	double val = reg.val;
	double err = reg.err;
	q.push(reg);

	while (abs(err / val) > eps)
	{
		if (31 * (int)q.size() >= maxCalls)
			throw numerics_exception(
			    "Gauss-Kronrod adaptive integral did not converge.");

		reg = q.top();
		q.pop();
		auto regLeft = Region(f, reg.a, 0.5 * (reg.a + reg.b));
		auto regRight = Region(f, 0.5 * (reg.a + reg.b), reg.b);
		val += regLeft.val + regRight.val - reg.val;
		err += regLeft.err + regRight.err - reg.err;
		q.push(regLeft);
		q.push(regRight);
	}
	return val;
}

double integrate(function_t f, double a, double b)
{
	return integrate(f, a, b, 1.0e-12, 5000);
}
