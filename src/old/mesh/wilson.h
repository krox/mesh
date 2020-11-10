#ifndef MESH_WILSON_H
#define MESH_WILSON_H

/** wilson loops on a regular lattice */

#include "mesh/mesh.h"

/**
 * Compute all planar na x nb loops with na <= nb <= maxN.
 * Averages over all positions and orientations
 */
template <typename G>
std::vector<std::tuple<int, int, double>> wilson(const Mesh<G> &m, int maxN);

#endif
