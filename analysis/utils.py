""" utilities to analyze results from the C++ simulations """

import numpy as np
import matplotlib.pyplot as plt
import h5py
from natsort import natsorted
from puck.fitting import *
import progressbar as pb
import itertools
from puck.fitting import *
import progressbar as pb

class UnionFind:
    """ simple 'disjoint-set' datastructure """

    __slots__ = ["_parent", "_size", "_nComp"]
    def __init__(self, n):
        self._parent = list(range(n))
        self._size = [1]*n
        self._nComp = n
    def root(self, a):
        assert 0 <= a < len(self._parent)
        while self._parent[a] != a:
            a = self._parent[a] # should do path compression here...
        return a
    def join(self, a, b):
        a = self.root(a)
        b = self.root(b)
        if a == b:
            return False
        if self._size[a] < self._size[b]:
            a,b = b,a
        self._parent[b] = a
        self._size[a] += self._size[b]
        self._nComp -= 1
        return True

    def is_joined(self, a, b):
        return self.root(a) == self.root(b)
    def comp_size(self, a):
        return self._size[self.root(a)]
    def __len__(self):
        return len(self._parent)
    def components(self):
        comp = [-1]*len(self)
        count = 0
        for i in range(len(self)):
            if self._parent[i] == i:
                comp[i] = count
                count += 1
        assert count == self._nComp
        for i in range(len(self)):
            comp[i] = comp[self.root(i)]
        return comp

def make_mom_list(mom2max, nd):
    """ list all integer momenta with p^2 <= mom2max """

    assert nd >= 0
    m = int(mom2max**0.5)
    full_list = itertools.product(*[list(range(-m,m+1))]*nd)
    return [p for p in full_list if sum([x**2 for x in p]) <= mom2max]

def make_phases(mom_list, geom):
    """ returns [moms, geom] array of phase factors """

    phases = []
    co = np.meshgrid(*[range(n) for n in geom], indexing="ij", sparse=True)
    for mom in mom_list:
        assert len(mom) == len(geom)
        tmp = 2.*np.pi*sum(mom[i]*co[i]/geom[i] for i in range(len(geom)))
        phases.append(np.cos(tmp) + 1.j*np.sin(tmp))
    return np.array(phases)

def try_get_correlator(filename, mom2max, path="c2pt"):
    """ same as get_correlator, but returns None if data is not already available """

    with h5py.File(filename, "r") as file:
        geom = file.attrs["geometry"]
        nc = file.attrs["markov_count"]
        mom_list = make_mom_list(mom2max=mom2max, nd=len(geom)-1)

        c2pt = np.zeros((nc, mom2max+1, geom[-1]))
        c2pt_count = np.zeros(mom2max+1)
        for mom in mom_list:
            mom2 = sum(x**2 for x in mom)
            name = f"{path}/{''.join(str(x) for x in mom)}_real"
            if name not in file:
                return None
            data = file[name][:]
            c2pt_count[mom2] += 1
            c2pt[:,mom2,:] += data[:,:]
    c2pt = c2pt[:, c2pt_count != 0, :]
    c2pt_count = c2pt_count[c2pt_count != 0]
    return c2pt / c2pt_count[np.newaxis, :, np.newaxis]


def get_correlator(filename, mom2max=6):
    """
    Compute 2-point correlator of an (Ising) ensemble
        * result is cached in the file itself
        * data is stored separately per momentum as "c2pt/001_real" and similar
        * returns [nc, mom^2, T] array
        * impossible mom^2 are skipped in the array (depends on dimension)
    """

    # try reading existing data
    c2pt = try_get_correlator(filename, mom2max=mom2max)
    if c2pt is not None:
        return c2pt

    # otherwise, compute correlator now. As a precaution against hdf5 file
    # corruption, we keep the file in read-only most of the time
    with h5py.File(filename, "r") as file:
        geom = file.attrs["geometry"]
        mom_list = make_mom_list(mom2max=mom2max, nd=len(geom)-1)
        config_list = natsorted(list(file["configs"]))

        print(f"computing correlator(s) for {filename}, nc={len(config_list)}, mom2max={mom2max}")

        c2pt = np.zeros((len(mom_list), len(config_list), geom[-1]))
        pb.streams.flush()
        for i, config in pb.progressbar(list(enumerate(config_list))):
            data = file["configs"][config][:].squeeze()
            corr = np.fft.ifft(np.abs(np.fft.fftn(data))**2, axis=-1) / data.size
            for momi, mom in enumerate(mom_list):
                c2pt[momi,i,:] = corr[mom].real
        pb.streams.flush()

    print(f"writing result into hdf5")
    with h5py.File(filename, "r+") as file:
        file.require_group("c2pt")
        for momi, mom in enumerate(mom_list):
            name = f"c2pt/{''.join(str(x) for x in mom)}_real"
            file.require_dataset(name, shape = c2pt[momi].shape, dtype=c2pt[momi].dtype, data=c2pt[momi], chunks=True, fletcher32=True)

    c2pt = try_get_correlator(filename, mom2max=mom2max)
    assert c2pt is not None
    return c2pt

def get_correlator_improved(filename, mom2max=0, seed=None):
    """
    same result as get_correlator(), but uses an improved estimator
       * current implementation uses slow spatial Fourier transform, so runtime
         increases linearly with number of momenta.
       * also current implementation is quadratic in volume, which could
         be improved. But maybe, a C++ implementation is worthwhile at that
         point.
       * compared to the basic estimator, noise is improved quite a bit for
         large distances. Though if the goal is just mass estimation, its not
         really worth the increased runtime. (As long as there are no excited
         states.)
    """

    # try reading existing data
    c2pt = try_get_correlator(filename, mom2max=mom2max, path="c2pt_improved")
    if c2pt is not None:
        return c2pt

    # NOTE: This implementation is quite slow ( O(volume * #clusters * log(...)) = O(volume^2 * log(...)) ).
    #       This can be solved by more careful analysis of cluster sizes and such, but that seems ugly (and
    #       possibly slow in python). So for the meantime, we optimize by using a slow Fourier transform in
    #       space, which is faster than an fft as long as only very few momenta are requested

    with h5py.File(filename, "r") as file:
        geom = file.attrs["geometry"] # geometry of the lattice
        beta = file.attrs["beta"] # (inverse) coupling
        top = file["topology"] # topology (graph) of the lattice
        config_list = natsorted(list(file["configs"]))
        nd = len(geom) # number of dimensions
        vol = np.product(geom) # total volume of the lattice
        p = 1.0 - np.exp(-2.0*beta) # link-probability
        rng = np.random.RandomState(seed=seed)

        print(f"computing (improved) correlator(s) for {filename}, nc={len(config_list)}, mom2max={mom2max}")

        # momenta and phase factors for (slow) spatial Fourier transform
        mom_list = make_mom_list(mom2max=mom2max, nd=nd-1) # spatial momenta
        # NOTE: "phases = phases[...,np.newaxis]" does not work because
        #       np.sum(a,where=w) only broadcast w to a, not the other way around.
        phases = make_phases(mom_list=mom_list, geom=geom[:-1])
        phases = np.broadcast_to(phases[...,np.newaxis], shape=(len(mom_list),*geom))

        # result
        c2pt = np.zeros((len(mom_list), len(config_list), geom[-1]))

        pb.streams.flush()
        for i, config in pb.progressbar(list(enumerate(config_list))):

            # create clusters
            field = file["configs"][config][:].squeeze().flatten()
            uf = UnionFind(vol)
            for a,b in top:
                if field[a] == field[b] and rng.rand() < p:
                    uf.join(a,b)
            clusters = np.array(uf.components()).reshape(geom)

            # contract clusters and (spatial) phases
            tmp = np.zeros((uf._nComp, len(mom_list), geom[-1]), dtype=np.complex128)
            for c in range(uf._nComp):
                tmp[c,:,:] = np.sum(phases, where=(clusters==c), axis=tuple(range(1,nd)))

            # compute correlation along time direction
            tmp = np.abs(np.fft.fft(tmp, axis=2))**2
            tmp = tmp.sum(axis=0)
            assert(tmp.ndim==2)
            c2pt[:, i, :] = np.fft.ifft(tmp,axis=1).real / clusters.size

    pb.streams.flush()

    print(f"writing result into hdf5")
    with h5py.File(filename, "r+") as file:
        file.require_group("c2pt")
        for momi, mom in enumerate(mom_list):
            name = f"c2pt_improved/{''.join(str(x) for x in mom)}_real"
            file.require_dataset(name, shape = c2pt[momi].shape, dtype=c2pt[momi].dtype, data=c2pt[momi], chunks=True, fletcher32=True)

    c2pt = try_get_correlator(filename, mom2max=mom2max, path="c2pt_improved")
    assert c2pt is not None
    return c2pt

def fit_mass_multi(data, geom, **kwargs):
    """ fit mass with multiple momenta simultaneously """

    assert data.ndim == 3
    nd = len(geom)
    L = geom[0] # assuming its square in space

    # list possible mom^2
    if nd == 2: p2_list = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    elif nd == 3: p2_list = [0, 1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25, 26, 29, 32, 34, 36, 37, 40, 41, 45, 49, 50, 52, 53, 58, 61, 64, 65, 68, 72, 73, 74, 80, 81, 82, 85, 89, 90, 97, 98]
    elif nd >= 4: p2_list = list(range(data.shape[1]))
    else: assert False
    assert len(p2_list) >= data.shape[1]
    p2_list = p2_list[:data.shape[1]]

    models = []
    for p2i, p2 in enumerate(p2_list):
        e = np.zeros((len(p2_list), 1))
        e[p2i] = 1.0
        models.append(lambda x, m, e=e, p2=(p2*(2*np.pi/L)**2): np.exp(-np.sqrt(m**2 + p2)*x)*e)

    if "label" not in kwargs:
        kwargs["label"] = [f"$p^2={p2}$" for p2 in p2_list]
    if "param_names" not in kwargs:
        kwargs["param_names"] = [f"c{p2}" for p2 in p2_list] + ["m"]

    return fit_varpro(data, models=models, guesses=[0.2], **kwargs)

def analyze_mass(filename, mom2max=6, binsize=None, improved=False, **kwargs):
    kwargs["plot_log"] = True
    if improved:
        c2pt = get_correlator_improved(filename, mom2max=mom2max)
    else:
        c2pt = get_correlator(filename, mom2max=mom2max)
    if binsize is not None:
        c2pt = binned(c2pt, binsize=binsize)
    with h5py.File(filename, "r") as file:
        geom = file.attrs["geometry"][:]

    c2pt = c2pt[:,:,:c2pt.shape[-1]//4]
    return fit_mass_multi(data=c2pt, geom=geom, **kwargs)
