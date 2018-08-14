#ifndef SCALAR_SCALAR_H
#define SCALAR_SCALAR_H

#include <cstdint>
#include <string>
#include <vector>

namespace scalar {

/** parameters of markov chain */
struct ChainParams
{
	/** parameters with physical meaning */
	std::vector<int> geom = {10000}; // size of lattice
	double kappa = 0;                // hopping parameter
	double lambda = 0;               // quartic coupling

	/** additional simulation parameters */
	int count = 100;   // number of configs to generate
	int discard = 0;   // number of discarded configs
	int sweeps = 1;    // number of sweeps between measurements
	int overrelax = 0; // number of OR steps per heat-bath sweeps

	uint64_t seed = 0;         // seed for random number generator
	std::string filename = ""; // empty if no output is required

	std::string autoFilename() const;
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

} // namespace scalar
#endif
