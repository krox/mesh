#ifndef GAUGE_GAUGE_H
#define GAUGE_GAUGE_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "xtensor/xarray.hpp"

#include "util/io.h"
#include "util/random.h"
#include "util/span.h"

/** helper class for topology. I.e. bunch of indices */
class GaugeTopology
{
  public:
	std::vector<int> geom;

	// plaqs: u(i) * u(j) * u(k).adj * u(l).adj
	// staples 0,1,2: u(j) * u(k).adj * u(l).adj
	//         3,4,5: u(j).adj * u(k).adj * u(l)
	std::vector<std::array<int, 4>> plaqs;
	std::vector<std::array<std::array<int, 3>, 6>> staples;

	GaugeTopology(const std::vector<int> &geom);

	int nSites() const
	{
		int s = 1;
		for (int n : geom)
			s *= n;
		return s;
	}

	int nLinks() const { return 4 * nSites(); }
	int nPlaqs() const { return 6 * nSites(); }
};

/** gauge field with a couple of basic helper functions. No specific action */
template <typename G> class GaugeMesh
{
  public:
	std::shared_ptr<const GaugeTopology> top;
	std::vector<G> u;

	int nSites() const { return top->nSites(); }
	int nLinks() const { return top->nLinks(); }
	int nPlaqs() const { return top->nPlaqs(); }

	/** initializes to unit-field */
	GaugeMesh(const std::shared_ptr<const GaugeTopology> &top)
	    : top(top), u(nLinks(), G::one())
	{}

	/** sum/average of staples of link i */
	G stapleSum(int i) const;
	G stapleAvg(int i) const { return stapleSum(i) * (1.0 / 6.0); };

	/** sum/average of all plaquettes */
	double plaqSum() const;
	double plaqAvg() const { return plaqSum() / nPlaqs(); }

	/** set all links to unit ("free field") */
	void initUnit()
	{
		for (auto &x : u)
			x = G::one();
	}

	/** set all links randomly ("weak field") */
	void initRandom(rng_t &rng)
	{
		for (auto &x : u)
			x = G::random(rng);
	}

	/** link smearing */
	GaugeMesh<G> smearAPE(double alpha) const;
	GaugeMesh<G> smearEXP(double alpha) const;
	// future: HYP smearing

	/** topological charge */
	double topCharge() const;

	span<const double> rawConfig() const
	{
		return span<const double>((double const *)u.data(),
		                          u.size() * G::repSize());
	}
};

#endif
