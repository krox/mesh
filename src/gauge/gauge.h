#ifndef GAUGE_GAUGE_H
#define GAUGE_GAUGE_H

#include <cassert>
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
	std::array<int, 4> geom;

	/** (x,mu) -> linkId.
	 Note: this takes nontrivial time (due to %), so caching is advisable */
	int lid(std::array<int, 4> x, int mu) const
	{
		x[0] = (x[0] + geom[0]) % geom[0];
		x[1] = (x[1] + geom[1]) % geom[1];
		x[2] = (x[2] + geom[2]) % geom[2];
		x[3] = (x[3] + geom[3]) % geom[3];
		int i = ((x[0] * geom[1] + x[1]) * geom[2] + x[2]) * geom[3] + x[3];
		return 4 * i + mu;
	}

	int lid(std::array<int, 4> x, int shift, int mu) const
	{
		x[shift] += 1;
		return lid(x, mu);
	}

	int lid(std::array<int, 4> x, int shift, int shift2, int mu) const
	{
		x[shift] += 1;
		x[shift2] += 1;
		return lid(x, mu);
	}

	// plaqs: u(i) * u(j) * u(k).adj * u(l).adj
	std::vector<std::array<int, 4>> plaqs; // size = 6 * nSites

	// rects: u(i) * u(j) * u(k) * u(l).adj * u(m).adj * u(n).adj
	std::vector<std::array<int, 6>> rects; // count = 24 * nSites

	// staples 0,1,2: u(j) * u(k).adj * u(l).adj
	//         3,4,5: u(j).adj * u(k).adj * u(l)
	std::vector<std::array<std::array<int, 3>, 6>> staples;

	// long staples 0-6: bcDEF, 6-11: bCDEf, 12-17: BCDef
	std::vector<std::array<std::array<int, 5>, 18>> lstaples;

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
	int nRects() const { return 12 * nSites(); }
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
	int nRects() const { return top->nRects(); }

	/** initializes to unit-field */
	GaugeMesh(const std::shared_ptr<const GaugeTopology> &top)
	    : top(top), u(nLinks(), G::one())
	{}

	/** sum/average of staples of link i */
	G stapleSum(int i) const;
	G stapleAvg(int i) const { return stapleSum(i) * (1.0 / 6.0); };
	G lstapleSum(int i) const;
	G lstapleAvg(int i) const { return lstapleSum(i) * (1.0 / 18.0); };

	/** sum/average of all plaquettes/rectangles */
	double plaqSum() const;
	double plaqAvg() const { return plaqSum() / nPlaqs(); }
	double rectSum() const;
	double rectAvg() const { return rectSum() / nRects(); }

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
	GaugeMesh<G> smearCool() const;
	GaugeMesh<G> smearCool(int count) const
	{
		assert(count >= 1);
		if (count == 1)
			return smearCool();
		else
			return smearCool().smearCool(count - 1);
	}
	GaugeMesh<G> smearAPE(double alpha) const;
	GaugeMesh<G> smearAPE(double alpha, int count) const
	{
		assert(count >= 1);
		if (count == 1)
			return smearAPE(alpha);
		else
			return smearAPE(alpha).smearAPE(alpha, count - 1);
	}

	GaugeMesh<G> smearEXP(double alpha) const;
	GaugeMesh<G> smearEXP(double alpha, int count) const
	{
		assert(count >= 1);
		if (count == 1)
			return smearEXP(alpha);
		else
			return smearEXP(alpha).smearEXP(alpha, count - 1);
	}
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
