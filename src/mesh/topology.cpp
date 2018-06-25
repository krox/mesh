#include "mesh/topology.h"

Topology Topology::lattice1D(int N)
{
	Topology top(N);
	top.top = "periodic1";
	top.geom = {N};
	top.links.reserve(top.nSites());

	for (int x = 0; x < N; ++x)
		top.addLink(x, (x + 1) % N);

	assert(top.nLinks() == top.nSites());
	return top;
}

Topology Topology::lattice2D(int N)
{
	Topology top(N * N);
	top.top = "periodic2";
	top.geom = {N, N};
	top.links.reserve(2 * top.nSites());

	auto f = [=](int x, int y) { return (x % N) * N + (y % N); };

	for (int x = 0; x < N; ++x)
		for (int y = 0; y < N; ++y)
		{
			top.addLink(f(x, y), f(x + 1, y));
			top.addLink(f(x, y), f(x, y + 1));
		}

	assert(top.nLinks() == 2 * top.nSites());
	return top;
}

Topology Topology::lattice3D(int N)
{
	Topology top(N * N * N);
	top.top = "periodic3";
	top.geom = {N, N, N};
	top.links.reserve(3 * top.nSites());

	auto f = [=](int x, int y, int z) {
		return ((x % N) * N + (y % N)) * N + (z % N);
	};

	for (int x = 0; x < N; ++x)
		for (int y = 0; y < N; ++y)
			for (int z = 0; z < N; ++z)
			{
				top.addLink(f(x, y, z), f(x + 1, y, z));
				top.addLink(f(x, y, z), f(x, y + 1, z));
				top.addLink(f(x, y, z), f(x, y, z + 1));
			}

	assert(top.nLinks() == 3 * top.nSites());
	return top;
}

Topology Topology::lattice4D(int N)
{
	Topology top(N * N * N * N);
	top.top = "periodic4";
	top.geom = {N, N, N, N};
	top.links.reserve(4 * top.nSites());

	auto f = [=](int x, int y, int z, int t) {
		return (((x % N) * N + (y % N)) * N + (z % N)) * N + (t % N);
	};

	for (int x = 0; x < N; ++x)
		for (int y = 0; y < N; ++y)
			for (int z = 0; z < N; ++z)
				for (int t = 0; t < N; ++t)
				{
					top.addLink(f(x, y, z, t), f(x + 1, y, z, t));
					top.addLink(f(x, y, z, t), f(x, y + 1, z, t));
					top.addLink(f(x, y, z, t), f(x, y, z + 1, t));
					top.addLink(f(x, y, z, t), f(x, y, z, t + 1));
				}

	assert(top.nLinks() == 4 * top.nSites());
	return top;
}
