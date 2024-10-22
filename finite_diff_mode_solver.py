import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spl


def get_eigen_matrix(L, k):
    """Computes k eigenvectors and eigenvalues of sparse matrix L.

    Parameters
    ----------
    L: sparse Matrix
    k: number of eigenvectors and eigenvalues to be returned

    Returns
    -------
    ev: eigenvalues
    vecs: eigenvectors vertically stacked together
    """

    # calculate eigenvalues and take real part of them since
    # absorption is negligible
    # eigsh seems to be faster than eigs
    # since our matrix L is symmetric, we can use it
    ev, vecs = spl.eigsh(L, k=k)
    ev = np.real(ev)
    vecs = np.real(vecs)

    # swap axis in vecs, because vecs are aligned horizontally
    vecs = vecs.swapaxes(1, 0)

    # sort the results according to the eigenvalues
    order = np.flip(ev.argsort())
    ev = ev[order]
    vecs = vecs[order]

    # since eigenvectors can be scaled by any constant, we decided to
    # normalize them with respect to their maximum value or their
    # minimum value depending which one is larger (abs)
    for i, v in enumerate(vecs):
        # due to symmetric we can search for the max/min in one half
        mi = np.min(v[0:len(v) // 2])
        ma = np.max(v[0:len(v) // 2])
        if abs(mi) > abs(ma):
            vecs[i] = v / mi
        else:
            vecs[i] = v / ma

    return ev, vecs


def guided_modes_1DTE(prm, k0, h, dtype_mat=np.float64):
    """Computes the effective permittivity of a TE polarized guided eigenmode.
    All space dimensions are in µm.
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

    # diagonal of sparse matrix
    den = 1 / k0 ** 2 / h ** 2
    diag = -2 * den + prm
    # off diagonals of the matrix
    n_diag = np.ones(prm.shape) * den

    # fill the sparse matrix with the data
    data = np.array([diag, n_diag, n_diag]).astype(dtype_mat)
    offset = np.array([0, 1, -1])
    L = sps.diags(data, offset)
    # call function to calculate normalize eigenvectors and eigenvalues
    ev, vecs = get_eigen_matrix(L, len(prm) - 2)

    # filter for the physical interesting
    # first we create a binary mask which checks for the conditions
    mask = np.logical_and(ev > prm[0], ev < np.max(prm))
    # extract these eigenvalues
    eff_eps = ev[mask]
    # also extract the eigenvectors
    vecs = vecs[mask]

    return eff_eps, vecs


def guided_modes_2D(prm, k0, h, numb, dtype_mat=np.float64):
    """Computes the effective permittivity of a quasi-TE polarized guided
    eigenmode.
    All space dimensions are in µm.

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
    # the output won't be necessarily symmetric!
    Ny = prm.shape[0]
    Nx = prm.shape[1]
    N = Ny * Nx
    # reshape prm to a 1d array
    prm = prm.flatten()

    # definitions used for the sparse matrix
    den = 1 / k0 ** 2 / h ** 2
    # diagonal of sparse matrix
    diag = (-4 * den + prm)
    # off diagonals of the matrix
    n_diag = np.ones(N) * den
    # this mask determines all position of zeros
    mask_n_diag = np.arange(1, N + 1) % Nx == 0
    # set all position to zero where the mask is true
    n_diag[mask_n_diag] = 0
    # the other diagonals
    other = np.ones(N) * den

    # fill the sparse matrix with the data
    data = np.array([diag, n_diag, n_diag, other, other]).astype(dtype_mat)
    offset = np.array([0, 1, -1, -Nx, Nx])
    L = sps.diags(data, offset)

    # get eigenvalues and eigenvectors
    ev, vecs = get_eigen_matrix(L, numb)

    n_vecs = []
    # we need to reshape the obtained vecs into a 2d array
    for v in vecs:
        n_vecs.append(v.reshape((Ny, Nx)))
    n_vecs = np.array(n_vecs)

    return ev, n_vecs
