from .direct import *
from . import direct
from . import sparse
from . import tridiagonal


def select_skewherm(flag):
    """
    Select whether matrices for Laplacian are skew Hermitian.

    Parameters
    ----------
    flag: bool

    Returns
    -------
    None
    """
    if flag:
        direct.solve_direct_ = solve_direct_skewh_
        direct.dot_direct_ = dot_direct_skewh_
    else:
        direct.solve_direct_ = solve_direct_nonskewh_
        direct.dot_direct_ = dot_direct_skewh_

    