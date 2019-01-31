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
	/** Memory consumption per site:
	SU(2) links:  128
	SU(3) links:  576

	plaqs:         96
	staples:      288
	clovers:      384
	rects:        288
	staples:     1440
	sum indices: 2496
	*/
  public:
	std::array<int, 4> geom;

	/** x -> siteId. This takes nontrivial time, so caching is advisable */
	int sid(std::array<int, 4> x) const
	{
		x[0] = (x[0] + geom[0]) % geom[0];
		x[1] = (x[1] + geom[1]) % geom[1];
		x[2] = (x[2] + geom[2]) % geom[2];
		x[3] = (x[3] + geom[3]) % geom[3];
		return ((x[0] * geom[1] + x[1]) * geom[2] + x[2]) * geom[3] + x[3];
	}

	int sid(std::array<int, 4> x, int shift) const
	{
		x[shift] += 1;
		return sid(x);
	}

	int sid(std::array<int, 4> x, int shift, int shift2) const
	{
		x[shift] += 1;
		x[shift2] += 1;
		return sid(x);
	}

	// plaqs: u(i) * u(j) * u(k).adj * u(l).adj
	std::vector<std::array<int, 4>> plaqs; // size = 6 * nSites

	// rects: u(i) * u(j) * u(k) * u(l).adj * u(m).adj * u(n).adj
	std::vector<std::array<int, 6>> rects; // count = 24 * nSites

	// 4 plaqs per orientation: abCD, aBCd, ABcd, AbcD
	std::vector<std::array<std::array<std::array<int, 4>, 4>, 6>> clovers;

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

	/** maximal distance from group */
	double error() const;

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

	/** sum of four plaquettes */
	G clover(int i, int o) const;

	/** topological charge */
	double topCharge() const;

	span<const double> rawConfig() const
	{
		return span<const double>((double const *)u.data(),
		                          u.size() * G::repSize());
	}

	span<double> rawConfigMut()
	{
		return span<double>((double *)u.data(), u.size() * G::repSize());
	}
};

#endif
