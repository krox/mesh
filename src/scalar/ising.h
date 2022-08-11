#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct IsingParams
{
	// physical parameters
	std::vector<int> geom = {32, 32};  // size of lattice
	double beta = 0.44068679350977147; // nearest-neighbour coupling

	// additional parameters for the Markov chain
	// (precise meaning might depend on the simulation algorithm used)
	int count = 100;       // number of configs to generate
	int discard = 0;       // number of discarded configs
	int spacing = 1;       // number of updates between configs
	std::string seed = ""; // seed for random number generator

	// output of configs
	std::string filename = "";
	bool overwrite_existing = false;
};

struct IsingResults
{
	std::vector<double> actionHistory;
	std::vector<double> magnetizationHistory;
	std::vector<double> susceptibilityHistory;
};

IsingResults runSwendsenWang(const IsingParams &params);
IsingResults runHeatBath(const IsingParams &params);
IsingResults runProppWilson(const IsingParams &params);
