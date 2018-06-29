#ifndef MESH_MARKOV_H
#define MESH_MARKOV_H

/** run the markov chain to create configs */

#include <cstdint>
#include <string>
#include <vector>

/** parameters of markov chain */
struct ChainParams
{
	/** parameters with physical meaning */
	std::string group = "su2";            // name of gauge group (i.e. "su2")
	std::vector<int> geom = {8, 8, 8, 8}; // size of lattice
	double beta = 1, beta2 = 0;           // inverse coupling

	/** additional simulation parameters */
	int init = 3; // initial condition of lattice (1=ordered, 2=random, 3=mixed)
	int count = 100;   // number of configs to generate
	int discard = 0;   // number of discarded configs
	int sweeps = 1;    // number of sweeps between measurements
	int overrelax = 0; // number of OR steps per heat-bath sweeps

	uint64_t seed = 0;         // seed for random number generator
	std::string filename = ""; // empty if no output is required
};

/** some measurements taken during the simulation. This may include measurements
 * on intermediate configs that where not saved. */
struct ChainResult
{
	double corrTime = 0.0 / 0.0; // correlation time in the resulting ensemble
	double action;               // normalized action (i.e. average plaquette)

	/** these include in-between configs but no thermalization */
	std::vector<double> actionHistory;
};

ChainResult runChain(const ChainParams &params);

#endif
