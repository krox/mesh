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

def make_mom_list(mom2max, nd):
    """ list all integer momenta with p^2 <= mom2max """

    assert nd >= 0
    m = int(mom2max**0.5)
    full_list = itertools.product(*[list(range(-m,m+1))]*nd)
    return [p for p in full_list if sum([x**2 for x in p]) <= mom2max]

def try_get_correlator(filename, mom2max=6):
    """ same as get_correlator, but returns None if data is not already available """

    with h5py.File(filename, "r") as file:
        geom = file.attrs["geometry"]
        nc = file.attrs["markov_count"]
        mom_list = make_mom_list(mom2max=mom2max, nd=len(geom)-1)

        c2pt = np.zeros((nc, mom2max+1, geom[-1]))
        c2pt_count = np.zeros(mom2max+1)
        for mom in mom_list:
            mom2 = sum(x**2 for x in mom)
            name = f"c2pt/{''.join(str(x) for x in mom)}_real"
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

def analyze_mass(filename, mom2max=6, binsize=None, **kwargs):
    kwargs["plot_log"] = True
    c2pt = get_correlator(filename, mom2max=mom2max)
    if binsize is not None:
        c2pt = binned(c2pt, binsize=binsize)
    with h5py.File(filename, "r") as file:
        geom = file.attrs["geometry"][:]

    c2pt = c2pt[:,:,:c2pt.shape[-1]//4]
    return fit_mass_multi(data=c2pt, geom=geom, **kwargs)
