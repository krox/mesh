#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "lattice/grid.h"
#include "lattice/laplace.h"
#include "util/gnuplot.h"

using namespace mesh;

int main(int argc, char **argv)
{
	std::vector<int32_t> geom = {50, 50};
	bool doPlot = false;
	double h = 0.02;

	CLI::App app{"Solve the (scalar) Poisson equation."};
	app.add_option("--geom", geom, "geometry of the lattice");
	app.add_option("--spacing", h, "lattice spacing (isotropic)");
	app.add_flag("--plot", doPlot, "do some plots summarizing the trajectory");
	CLI11_PARSE(app, argc, argv);

	auto op = LaplaceOperator(h);

	assert(geom.size() == 2);
	auto &grid = Grid::make_double(Coordinate(geom.begin(), geom.end()));

	// create the "f" and "g" vectors
	auto g_fun = [](double x, double y) { return +40. + 0 * x + 0 * y; };
	auto f_fun = [](double x, double y) {
		return sin(x * 2 * M_PI) + cos(y * 2 * M_PI);
	};
	auto g = Lattice<util::simd<double>>(grid);
	auto f = Lattice<util::simd<double>>(grid);
	for (int i = 0; i < geom[0]; ++i)
		for (int j = 0; j < geom[1]; ++j)
		{
			g.pokeSite({i, j}, g_fun(i * h, j * h));
			f.pokeSite({i, j}, f_fun(i * h, j * h));
		}

	// solve the equation
	// in the interior: -D^2 u = g
	// on the boundary:      u = f
	auto u = Lattice<util::simd<double>>::zeros(grid);

	for (int iter = 0; iter < 200; ++iter)
	{
		u += op.apply_diagonal_inverse(g - op.apply(u));
		for (int i = 0; i < geom[0]; ++i)
			for (int j = 0; j < geom[1]; ++j)
				if (i == 0 || j == 0)
					u.pokeSite({i, j}, f.peekSite({i, j}));
	}

	if (doPlot)
	{
		auto result =
		    util::ndarray<double, 2>({(size_t)geom[0], (size_t)geom[1]});
		for (int i = 0; i < geom[0]; ++i)
			for (int j = 0; j < geom[1]; ++j)
				result(i, j) = u.peekSite({i, j});
		util::Gnuplot().style("lines").plotData3D(result());
	}
}
