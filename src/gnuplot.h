#ifndef MESH_GNUPLOT_H
#define MESH_GNUPLOT_H

#include "boost/format.hpp"
#include "stats.h"

class Gnuplot
{
	FILE *pipe = nullptr;
	int nplots = 0;
	using format = boost::format;

  public:
	std::string style = "points";

	/** constructor */
	explicit Gnuplot(bool persist = true);

	/** send a command to gnuplot */
	void cmd(const std::string &s);
	void cmd(const boost::basic_format<char> &fmt);

	/** plot a function given by a string that gnuplot can understand */
	void plotFunction(const std::string &fun, const std::string &title = "");

	/** plot raw data points (xs[i], ys[i]) */
	void plotData(const std::vector<double> &xs, const std::vector<double> &ys,
	              const std::string &title = "data");

	/** plot a histogram */
	void plotHistogram(const histogram &hist,
	                   const std::string &title = "hist");

	/** set range of plot */
	void setRangeX(double min, double max);
	void setRangeY(double min, double max);
	void setRangeZ(double min, double max);

	/** make the plot logarithmic */
	void setLogScaleX();
	void setLogScaleY();
	void setLogScaleZ();

	/** remove all plots (but keep settings) */
	void clear();
};

#endif
