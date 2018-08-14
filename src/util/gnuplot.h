#ifndef MESH_GNUPLOT_H
#define MESH_GNUPLOT_H

#include <functional>

#include "util/span.h"
#include "util/stats.h"

class Gnuplot
{
	FILE *pipe = nullptr;
	static int nplotsGlobal;
	int nplots = 0;
	const int plotID;

  public:
	std::string style = "points";

	/** constructor */
	explicit Gnuplot(bool persist = true);

	/** plot a function given by a string that gnuplot can understand */
	Gnuplot &plotFunction(const std::string &fun,
	                      const std::string &title = "");

	/** plot a function given as std::function object */
	Gnuplot &plotFunction(const std::function<double(double)> &fun, double min,
	                      double max, const std::string &title = "");

	/** plot raw data points (i, ys[i]) */
	Gnuplot &plotData(span<const double> ys, const std::string &title = "data");

	/** plot raw data points (xs[i], ys[i]) */
	Gnuplot &plotData(span<const double> xs, span<const double> ys,
	                  const std::string &title = "data");

	/** plot a histogram */
	Gnuplot &plotHistogram(const histogram &hist,
	                       const std::string &title = "hist");

	/** set range of plot */
	Gnuplot &setRangeX(double min, double max);
	Gnuplot &setRangeY(double min, double max);
	Gnuplot &setRangeZ(double min, double max);

	/** make the plot logarithmic */
	Gnuplot &setLogScaleX();
	Gnuplot &setLogScaleY();
	Gnuplot &setLogScaleZ();

	/** remove all plots (but keep settings) */
	Gnuplot &clear();
};

#endif
