from .transforms import as_fun
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from mpl_toolkits.axes_grid1 import ImageGrid


def plot(data, ax=None, symmetric=False, colorbar=True, use_ticks=True,
         xlabel="azimuth", ylabel="elevation",
         axes_pad=0.15, cbar_mode="edge", cbar_size="3%", **kwargs):
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
    use_ticks: bool
        Whether to use ticks on axes.
    xlabel: str
        Label on x-axis.
    ylabel: str
        Label on y-axis.
    axes_pad: float or (float, float)
        Padding in inches between axes. If tuple, then horizontal and vertical.
    cbar_mode: str
        Colorbar mode to send into `ImageGrid` (see its documentation for options).
    cbar_size:
        Colorbar size to send into `ImageGrid` (see its documentation for options).
    kwargs:
        Arguments to send to `ax.imshow(...)`.

    Returns
    -------
        Object returned by `ax.imshow(...)`.
    """
    if isinstance(data, tuple) or isinstance(data, list):
        if isinstance(data[0], tuple) or isinstance(data[0], list):
            ndimorig = 4
        else:
            data = [data]
            ndimorig = 3
        data = np.asarray(data)

        # Convert data to f-values
        fdata = []
        for di in data:
            fidata = []
            for d in di:
                fidata.append(as_fun(d))
            fdata.append(fidata)
        fdata = np.asarray(fdata)
        rows = fdata.shape[0]
        cols = fdata.shape[1]

        if ax is None:

            # Set up figure and image grid
            fig = plt.figure()

            # Compute padding
            if colorbar and cbar_mode == 'each':
                if not isinstance(axes_pad, tuple):
                    axes_pad = (axes_pad+0.4, axes_pad)
                cbar_pad = axes_pad[-1]/2
            # elif cbar_mode == 'single':
            #     cbar_pad = (axes_pad[0] if isinstance(axes_pad, tuple) else axes_pad) + 0.4
            else:
                cbar_pad = axes_pad[0] if isinstance(axes_pad, tuple) else axes_pad

            if colorbar and cbar_mode == 'single' and rows >= 2 and cbar_size == "3%":
                cbar_size = "1.5%"

            # Create image grid
            grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                             nrows_ncols=(rows, cols),
                             axes_pad=axes_pad,
                             share_all=True,
                             cbar_location="right",
                             cbar_mode=None if not colorbar else cbar_mode,
                             cbar_size=cbar_size,
                             cbar_pad=cbar_pad,
                             )

            axs = []
            for a in grid:
                axs.append(a)
            axs = np.array(axs).reshape((rows, cols))
        else:
            fig = None
            grid = None
            axs = np.array(ax)
            assert axs.shape == (rows, cols), "Number of data and axes elements must agree."

        # Compute limits if needed
        if 'vmin' in kwargs or 'vmax' in kwargs:
            set_minmax = False
        else:
            set_minmax = True
        if colorbar and cbar_mode == "single" and set_minmax:
            minmax = np.abs(fdata[0, 0]).max()  # Base values on first image
            kwargs['vmin'] = -minmax
            kwargs['vmax'] = minmax

        ims = []
        for i in range(rows):
            # Compute limits if needed
            if colorbar and cbar_mode == "edge" and set_minmax:
                minmax = np.abs(fdata[i, 0]).max()  # Base values on left-most image
                kwargs['vmin'] = -minmax
                kwargs['vmax'] = minmax

            for j in range(cols):
                f = fdata[i, j]
                ax = axs[i, j]

                # Do the plot
                ims.append(plot(f, ax=ax, xlabel=xlabel, ylabel=ylabel, symmetric=symmetric, use_ticks=use_ticks,
                                colorbar=False,
                                **kwargs))

                if colorbar and cbar_mode == "each":
                    cbar = ax.figure.colorbar(ims[-1], cax=ax.cax)

            if colorbar and cbar_mode == "edge":
                cbar = ax.figure.colorbar(ims[-1], cax=ax.cax)

        if colorbar and cbar_mode == "single" and grid is not None:
            grid.cbar_axes[0].colorbar(ims[0])

        # if fig is not None:
        #     fig.tight_layout(h_pad=0, w_pad=0)
        return np.asarray(ims).reshape((rows, cols))

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
        fig, ax = subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_label_coords(0.5, -.04)
    ax.yaxis.set_label_coords(-0.02, 0.5)
    if use_ticks:
        if symmetric:
            ax.set_xticks([-np.pi, np.pi])
            ax.set_xticklabels([r'$-\pi$', r'$\pi$'])
        else:
            ax.set_xticks([0, 2*np.pi])
            ax.set_xticklabels(['0', r'2$\pi$'])
        ax.set_yticks([0, np.pi])
        ax.set_yticklabels(['0', r'$\pi$'])
    else:
        ax.set_yticks([])
        ax.set_xticks([])

    if symmetric:
        fun = np.roll(fun, fun.shape[1]//2, axis=1)
    im = ax.imshow(fun, **kwargs)

    if colorbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        ax.figure.colorbar(im, cax=cax)

    return im
