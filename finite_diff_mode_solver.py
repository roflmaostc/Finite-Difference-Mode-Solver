'''Homework 1, Computational Photonics, SS 2020:  FD mode solver.
'''
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import eigs


def guided_modes_1DTE(prm, k0, h):
    """Computes the effective permittivity of a TE polarized guided eigenmode.
    All dimensions are in µm.
    Note that modes are filtered to match the requirement that
    their effective permittivity is larger than the substrate (cladding).

    Parameters
    ----------
    prm : 1d-array
        Dielectric permittivity in the x-direction
    k0 : float
        Free space wavenumber
    h : float
        Spatial discretization

    Returns
    -------
    eff_eps : 1d-array
        Effective permittivity vector of calculated modes
    guided : 2d-array
        Field distributions of the guided eigenmodes
    """
    out_shape = (len(prm), len(prm))

    # diagonal of sparse matrix
    den = 1 / k0 ** 2 / h ** 2
    diag = -2 * den + prm
    # of diagonals of the matrix
    diag_above = np.ones(prm.shape) * 1 * den
    diag_below = diag_above

    # fill the sparse matrix with the data
    data = np.array([diag, diag_above, diag_below])
    offset = np.array([0, 1, -1])
    L = sps.dia_matrix((data, offset), shape=out_shape)

    # calculate eigenvalues and take real part of them
    # absorption is negligibe
    ev, vecs = eigs(L, k=len(prm) - 2)
    ev = np.real(ev)
    vecs = np.real(vecs)
    # swap axis in vecs, because vecs are aligned horizontally
    vecs = vecs.swapaxes(1, 0)

    # filter for the physical interesting
    # first we create a binary mask which checks for the conditions
    mask = np.logical_and(ev > prm[0], ev < np.max(prm))

    # extract these eigenvalues
    eff_eps = ev[mask]
    # also extract the eigenvectors
    vecs = vecs[mask]

    return eff_eps, vecs

def guided_modes_2D(prm, k0, h, numb):
    """Computes the effective permittivity of a quasi-TE polarized guided 
    eigenmode. All dimensions are in µm.
    
    Parameters
    ----------
    prm  : 2d-array
        Dielectric permittivity in the xy-plane
    k0 : float
        Free space wavenumber
    h : float
        Spatial discretization
    numb : int
        Number of eigenmodes to be calculated
    
    Returns
    -------
    eff_eps : 1d-array
        Effective permittivity vector of calculated eigenmodes
    guided : 3d-array
        Field distributions of the guided eigenmodes
    """
    pass
