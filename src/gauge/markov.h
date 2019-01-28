#ifndef GAUGE_MARKOV_H
#define GAUGE_MARKOV_H

#include <string>
#include <vector>

#include "xtensor/xarray.hpp"

#include "gauge/wilson.h"

/** parameters of markov chain */
struct GaugeChainParams
{
	/** physical parameters */
	std::string group = "su2";            // z2, u1, su2
	std::vector<int> geom = {4, 4, 4, 4}; // size of lattice

	/** parameters of Markov chain */
	int count = 100;   // number of configs to generate
	int discard = 0;   // number of discarded configs
	int sweeps = 1;    // number of HB-sweeps between measurements
	int clusters = 0;  // number of cluster-flips per HB sweeps
	uint64_t seed = 0; // seed for random number generator

	/** technical parameters */
	std::string filename = "";
	bool skipConfig = false;
};

/** some measurements taken during the simulation */
struct GaugeChainResult
{
	xt::xarray<double> plaqHistory; // average plaquette
};

GaugeChainResult runChain(const GaugeChainParams &chainParams,
                          const WilsonActionParams &actionParams);

#endif
