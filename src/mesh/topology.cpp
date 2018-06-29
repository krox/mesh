#include "mesh/topology.h"

Topology Topology::lattice1D(int nx)
{
	Topology top(nx);
	top.top = "periodic1";
	top.geom = {nx};
	top.links.reserve(top.nSites());

	for (int x = 0; x < nx; ++x)
		top.addLink(x, (x + 1) % nx);

	assert(top.nLinks() == top.nSites());
	return top;
}

Topology Topology::lattice2D(int nx, int ny)
{
	Topology top(nx * ny);
	top.top = "periodic2";
	top.geom = {nx, ny};
	top.links.reserve(2 * top.nSites());

	auto f = [=](int x, int y) { return (x % nx) * ny + (y % ny); };

	for (int x = 0; x < nx; ++x)
		for (int y = 0; y < ny; ++y)
		{
			top.addLink(f(x, y), f(x + 1, y));
			top.addLink(f(x, y), f(x, y + 1));
		}

	assert(top.nLinks() == 2 * top.nSites());
	return top;
}

Topology Topology::lattice3D(int nx, int ny, int nz)
{
	Topology top(nx * ny * nz);
	top.top = "periodic3";
	top.geom = {nx, ny, nz};
	top.links.reserve(3 * top.nSites());

	auto f = [=](int x, int y, int z) {
		return ((x % nx) * ny + (y % ny)) * nz + (z % nz);
	};

	for (int x = 0; x < nx; ++x)
		for (int y = 0; y < ny; ++y)
			for (int z = 0; z < nz; ++z)
			{
				top.addLink(f(x, y, z), f(x + 1, y, z));
				top.addLink(f(x, y, z), f(x, y + 1, z));
				top.addLink(f(x, y, z), f(x, y, z + 1));
			}

	assert(top.nLinks() == 3 * top.nSites());
	return top;
}

Topology Topology::lattice4D(int nx, int ny, int nz, int nt)
{
	Topology top(nx * ny * nz * nt);
	top.top = "periodic4";
	top.geom = {nx, ny, nz, nt};
	top.links.reserve(4 * top.nSites());

	auto f = [=](int x, int y, int z, int t) {
		return (((x % nx) * ny + (y % ny)) * nz + (z % nz)) * nt + (t % nt);
	};

	for (int x = 0; x < nx; ++x)
		for (int y = 0; y < ny; ++y)
			for (int z = 0; z < nz; ++z)
				for (int t = 0; t < nt; ++t)
				{
					top.addLink(f(x, y, z, t), f(x + 1, y, z, t));
					top.addLink(f(x, y, z, t), f(x, y + 1, z, t));
					top.addLink(f(x, y, z, t), f(x, y, z + 1, t));
					top.addLink(f(x, y, z, t), f(x, y, z, t + 1));
				}

	assert(top.nLinks() == 4 * top.nSites());
	return top;
}

Topology Topology::lattice(const std::vector<int> &n)
{
	if (n.size() == 1)
		return lattice1D(n[0]);
	if (n.size() == 2)
		return lattice2D(n[0], n[1]);
	if (n.size() == 3)
		return lattice3D(n[0], n[1], n[2]);
	if (n.size() == 4)
		return lattice4D(n[0], n[1], n[2], n[3]);
	else
		throw std::runtime_error("invalid lattice dimension");
}
