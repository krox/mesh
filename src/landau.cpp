#include "lattice/landau.h"
#include "CLI/CLI.hpp"
#include "fmt/format.h"
#include "lattice/gauge.h"

using namespace mesh;

int main(int argc, char **argv)
{
	std::string group = "su3";
	std::string configName = "";
	int precision = 2;

	CLI::App app{"Perform Landau gauge fixing."};
	app.add_option("--group", group, "u1/su2/su3");
	app.add_option("--precision", precision);
	app.add_option("input", configName);

	CLI11_PARSE(app, argc, argv);

	dispatchByGroup(
	    [&]<typename vG>() {
		    auto U = readConfig<vG>(configName);
		    auto const &g = U.grid();
		    auto landau = Landau(U);
		    landau.verbose = true;
		    landau.run(10000);
	    },
	    group, precision);
}
