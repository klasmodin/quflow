from .transforms import as_fun
import numpy as np
import matplotlib.pyplot as plt


def plot(data, ax=None, symmetric=False, colorbar=True, **kwargs):
    """
    Plot quantized function. Good colormap arguments:
    - `cmap='twilight_shifted'` or `cmap = 'seismic'` for bright plots (this is default)
    - `cmap='twilight'` for dark plots

    Parameters
    ----------
    data: ndarray or tuple of ndarray
        Can be either mat, omegac, omegar, or fun.
    ax:
        Matplotlib axis to plot it (created if `None` which is default).
    symmetric: bool
        Use interval [-pi, pi] instead of [0, 2*pi] for `phi`.
        Defaults to `False`.
    colorbar: bool
        Whether to add colorbar to plot.
    kwargs:
        Arguments to send to `ax.imshow(...)`.

    Returns
    -------
        Object returned by `ax.imshow(...)`.
    """
    if isinstance(data, tuple) or isinstance(data, list):
        if ax is None:
            fig, axs = plt.subplots(len(data), sharex=True)
            fig.tight_layout(h_pad=0)
        else:
            assert len(ax) == len(data), "Number of data and axes elements must agree."
            axs = ax
        ims = []
        for (d, ax) in zip(data, axs):
            ax.set_ylabel(r'$\theta$', fontsize=16)
            ax.set_yticks([0, np.pi])
            ax.set_yticklabels(['0', r'$\pi$'])
            ims.append(plot(d, ax=ax, symmetric=symmetric, colorbar=colorbar, **kwargs))
        axs[-1].set_xlabel(r'$\phi$', fontsize=16)
        if symmetric:
            axs[-1].set_xticks([-np.pi, 0, np.pi])
            axs[-1].set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        else:
            axs[-1].set_xticks([0, np.pi, 2*np.pi])
            axs[-1].set_xticklabels(['0', r'$\pi$', r'2$\pi$'])
        return ims

    fun = as_fun(data)
    if np.iscomplexobj(fun):
        fun = fun.real
    minmax = np.abs(fun).max()
    if 'vmin' not in kwargs:
        kwargs['vmin'] = -minmax
    if 'vmax' not in kwargs:
        kwargs['vmax'] = minmax
    if 'extent' not in kwargs:
        if symmetric:
            kwargs['extent'] = [-np.pi, np.pi, np.pi, 0]
        else:
            kwargs['extent'] = [0, 2*np.pi, np.pi, 0]
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'seismic'
    if ax is None:
        from matplotlib.pyplot import subplots
        fig, ax = subplots()
        ax.set_xlabel(r'$\phi$', fontsize=16)
        if symmetric:
            ax.set_xticks([-np.pi, 0, np.pi])
            ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        else:
            ax.set_xticks([0, np.pi, 2*np.pi])
            ax.set_xticklabels(['0', r'$\pi$', r'2$\pi$'])
        ax.set_ylabel(r'$\theta$', fontsize=16)
        ax.set_yticks([0, np.pi])
        ax.set_yticklabels(['0', r'$\pi$'])

    if symmetric:
        fun = np.roll(fun, fun.shape[1]//2, axis=1)
    im = ax.imshow(fun, **kwargs)

    if colorbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        ax.figure.colorbar(im, cax=cax)

    return im
