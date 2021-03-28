from matplotlib.pyplot import imshow, subplots
from .transforms import as_fun
import numpy as np


def plot(data, ax=None, **kwargs):
    """
    Plot quantized function.

    Parameters
    ----------
    data: ndarray
        Can be either mat, omegac, omegar, or fun.
    ax:
        Matplotlib axis to plot it (created if `None` which is default).
    kwargs:
        Arguments to send to `ax.imshow(...)`.

    Returns
    -------
        Object returned by `ax.imshow(...)`.
    """
    fun = as_fun(data)
    if np.iscomplexobj(fun):
        fun = fun.real
    minmax = np.abs(fun).max()
    if 'vmin' not in kwargs:
        kwargs['vmin'] = -minmax
    if 'vmax' not in kwargs:
        kwargs['vmax'] = minmax
    if 'extent' not in kwargs:
        kwargs['extent'] = [0, 2*np.pi, np.pi, 0]
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'seismic'
    if ax is None:
        fig, ax = subplots()
        ax.set_xlabel(r'$\phi$', fontsize=16)
        ax.set_xticks([0, np.pi, 2*np.pi])
        ax.set_xticklabels(['0', r'$\pi$', r'2$\pi$'])
        ax.set_ylabel(r'$\theta$', fontsize=16)
        ax.set_yticks([0, np.pi])
        ax.set_yticklabels(['0', r'$\pi$'])

    im = ax.imshow(fun, **kwargs)
    return im
