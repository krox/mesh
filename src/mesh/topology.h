#pragma once

#include <array>
#include <cassert>
#include <string>
#include <vector>

/**
 * The 'Topology' class only contains the graph of the mesh, not the field.
 * Also it should be usable for both scalar fields and gauge fields.
 */

struct LinkRef
{
	int i : 31; // link id
	bool sign : 1;

	LinkRef() = default;
	LinkRef(int i, bool sign) : i(i), sign(sign) {}
};

struct HalfLink
{
	int to;
	LinkRef link;

	HalfLink() = default;
	HalfLink(int to, LinkRef link) : to(to), link(link) {}
	HalfLink(int to, int i, bool sign) : to(to), link(i, sign) {}
};

struct Link
{
	int from, to; // lattice sites this link is attached to

	Link() = default;
	Link(int from, int to) : from(from), to(to) {}
};

class Topology
{
  public:
	/* The pure graph itself */
	std::vector<Link> links;
	std::vector<std::vector<HalfLink>> graph;

	/* succesor site in time-direction
	  ( Only used for some imaginary action stuff.
	    only set for regular lattices. ) */
	std::vector<int> timeStep;

	/** type of topology (for regular lattices only) */
	std::string top;
	std::vector<int> geom;

	/** constructors */
	Topology() = default;
	explicit Topology(int nSites) : graph(nSites), timeStep(nSites, -1) {}

	/** create periodic, rectangular lattice topology */
	static Topology lattice1D(int nx);
	static Topology lattice2D(int nx, int ny);
	static Topology lattice3D(int nx, int ny, int nz);
	static Topology lattice4D(int nx, int ny, int nz, int nt);
	static Topology lattice(const std::vector<int> &n);

	/** number of sites/links */
	int nSites() const { return (int)graph.size(); }
	int nLinks() const { return (int)links.size(); }

	/** add a new site/link */
	int addSite();
	int addLink(int a, int b);

	/** list staples of a link */
	std::vector<std::array<LinkRef, 3>> staples4(int i) const;

	/** list all loops */
	std::vector<std::array<LinkRef, 4>> loops4() const;
};

inline int Topology::addSite()
{
	int i = nSites();
	graph.emplace_back();
	return i;
}

inline int Topology::addLink(int a, int b)
{
	int i = nLinks();
	links.emplace_back(a, b);
	graph[a].emplace_back(b, i, false);
	graph[b].emplace_back(a, i, true);
	return i;
}

inline std::vector<std::array<LinkRef, 3>> Topology::staples4(int i) const
{
	std::vector<std::array<LinkRef, 3>> staples;
	int a = links[i].from;
	int b = links[i].to;
	for (auto [c, j] : graph[b])
		if (c != a && c != b)
			for (auto [d, k] : graph[c])
				if (d != a && d != b && d != c)
					for (auto [e, l] : graph[d])
						if (e == a)
							staples.push_back({j, k, l});
	return staples;
}

inline std::vector<std::array<LinkRef, 4>> Topology::loops4() const
{
	std::vector<std::array<LinkRef, 4>> loops;
	for (int a = 0; a < nSites(); ++a)
		for (auto [b, i] : graph[a])
			if (b > a)
				for (auto [c, j] : graph[b])
					if (c > a && c != b)
						for (auto [d, k] : graph[c])
							if (d != c && d > b)
								for (auto [e, l] : graph[d])
									if (a == e) // loop closed?
										loops.push_back({i, j, k, l});
	return loops;
}
