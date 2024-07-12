from .transforms import as_fun, mat2shr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import subplots
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.animation as anim
from matplotlib.colors import hsv_to_rgb

try:
    import cartopy.crs as ccrs
    _has_cartopy = True
except ModuleNotFoundError:
    _has_cartopy = False


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def resample(data, N):
    """
    Up- or downsample data to resolution N.

    Parameters
    ----------
    data: ndarray, shape=(M,M) or shape=(M**2,)
    N: int
        New resolution.

    Returns
    -------
    omega with new resolution
    """
    if data.ndim == 2:
        if data.shape[0] == data.shape[1]:
            omega = mat2shr(data)
        else:
            raise NotImplementedError("Resampling fun data is not supported yet.")
    elif data.ndim == 1:
        omega = data
    omega2 = np.zeros(N**2, dtype=omega.dtype)
    omega2[:min(N**2,omega.shape[0])] = omega[:min(N**2,omega.shape[0])]
    return omega2


def create_animation(filename, states, fps=25, preset='medium', extra_args=[], codec='h264',
                     title='quflow simulation',
                     scale=None, N=None, **kwargs):
    """

    Parameters
    ----------
    filename
    states
    fps
    preset
    extra_args
    codec: str (default:'h264')
        ffmpeg codec. For accelerated Apple encoder, use 'h264_videotoolbox'
    title
    scale
    N: int or None (default None)
        Up- or downsample to resolution N in plot.
    kwargs

    Returns
    -------

    """

    FFMpegWriter = anim.writers['ffmpeg']
    metadata = dict(title=title, artist='Matplotlib', comment='http://github.com/klasmodin/quflow')
    extra_args = []

    if preset == 'medium':
        if '-b:v' not in extra_args:
            extra_args += ['-b:v', '3000K']
        if '-preset' not in extra_args and codec == 'h264':
            extra_args += ['-preset', 'veryslow']
        if scale is None:
            scale = 1
    elif preset == "low":
        if '-b:v' not in extra_args:
            extra_args += ['-b:v', '1500K']
        if scale is None:
            scale = 0.5
    elif preset == "high":
        if '-preset' not in extra_args and codec == 'h264':
            extra_args += ['-preset', 'veryslow']

    # Make sure some scale is selected
    if scale is None:
        scale = 1

    # Create ffmpeg writer
    writer = FFMpegWriter(fps=fps, metadata=metadata, codec=codec, extra_args=extra_args)

    dpi = 100

    if N is not None:
        omega = resample(states[0], N)
    else:
        omega = states[0]
    f0 = as_fun(omega)
    figsize = (f0.shape[1]/float(dpi), f0.shape[0]/float(dpi))

    with matplotlib.rc_context({'backend': 'Agg'}):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        # Hide spines, ticks, etc.
        ax.axis('off')

        im = plot(f0, ax=ax, colorbar=False, **kwargs) #, interpolation='nearest')
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(None)
        ax.set_ylabel(None)

        print("Writing file {}".format(filename))
        with writer.saving(fig, filename, dpi=dpi*scale):
            ndots = 40
            print("_"*min(ndots, states.shape[0]))
            for k in range(states.shape[0]):
                if N is not None:
                    omega = resample(states[k], N)
                else:
                    omega = states[k]
                fun = as_fun(omega)
                # TODO: insert code here for saving img if state file is writable
                if hasattr(im, 'set_data'):
                    im.set_data(fun)
                elif hasattr(im, 'set_array'):
                    im.set_array(fun)
                else:
                    raise NotImplementedError("Could not find method for setting data.")
                writer.grab_frame()
                if k % (max(states.shape[0], ndots)//ndots) == 0:
                    print("*", end='')
            print("")
        # Close figure (so it doesn't show up interactively)
        plt.close(fig=fig)

    # TODO: Return HTML displaying the movie
    if in_notebook():
        if False:
            from IPython.display import display, HTML
            htmlstr = "<div align=\"left\">"
            htmlstr += "<video width=\"{}%\" controls>".format(str(50))
            htmlstr += "<source src=\"{}\" type=\"video/mp4\">".format(filename)
            htmlstr += "</video></div>"
            htmlmovie = HTML(htmlstr)
            return display(htmlmovie)
        else:
            from IPython.display import Video
            return Video(filename, embed=False)
    else:
        print("Finished!")


def plot(data, ax=None, symmetric=False, colorbar=True, use_ticks=True,
         xlabel="azimuth", ylabel="elevation",
         axes_pad=0.15, cbar_mode="edge", cbar_size="3%", N=None, projection=None, **kwargs):
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
    N: int or None (default None)
        Up- or downsample to resolution N in plot.
    projection: None or str
        Which projection to use.
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
            if projection is None:
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
                axs = fig.subplots(nrows=rows, ncols=cols, subplot_kw={'projection': projection},
                                   gridspec_kw={'left': 0, 'right': 1, 'wspace': 0.02, 'hspace': 0.02})
                axs = axs.reshape((rows, cols))
                # fig.tight_layout(h_pad=0)
                # fig.set_dpi(150.) # Is this needed?

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
                                colorbar=False, projection=projection,
                                **kwargs))

                if colorbar and cbar_mode == "each":
                    cbar = ax.figure.colorbar(ims[-1], cax=ax.cax)

            if colorbar and cbar_mode == "edge":
                if projection is None:
                    cbar = ax.figure.colorbar(ims[-1], cax=ax.cax)
                else:
                    cbar = ax.figure.colorbar(ims[-1], ax=axs[:, -1], shrink=1.0)

        if colorbar and cbar_mode == "single" and grid is not None:
            grid.cbar_axes[0].colorbar(ims[0])

        # if fig is not None:
        #     fig.tight_layout(h_pad=0, w_pad=0)
        return np.asarray(ims).reshape((rows, cols))

    if N is not None:
        data = resample(data, N)
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
    if projection is not None:
        use_pcolormesh = True
    else:
        use_pcolormesh = False
    if ax is None:
        fig, ax = subplots(subplot_kw={'projection': projection})
    if not (use_pcolormesh and xlabel == 'azimuth'):
        ax.set_xlabel(xlabel)
    if not (use_pcolormesh and ylabel == 'elevation'):
        ax.set_ylabel(ylabel)

    if not use_pcolormesh:
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

    if use_pcolormesh:
        colorbar = False # Forcing no colorbar. TODO: This is a hack, should be fixed.
        lon = np.linspace(-np.pi, np.pi, fun.shape[1])
        lat = np.linspace(-np.pi/2., np.pi/2., fun.shape[0])
        Lon, Lat = np.meshgrid(lon, lat)
        if 'extent' in kwargs:
            kwargs.pop('extent')
        im = ax.pcolormesh(Lon, Lat, fun, rasterized=True, **kwargs)
        ax.grid(linestyle='-', color='black', alpha=0.2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        im = ax.imshow(fun, **kwargs)

    if colorbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        ax.figure.colorbar(im, cax=cax)

    return im


def plot2(data, ax=None, projection='hammer', dpi=None, gridon=True, colorbar=False, title=None,
          xlabel="azimuth", ylabel="elevation", padding=None, N=None, time=None,
          central_latitude=20, central_longitude=30, **kwargs):
    """
    Better plot quantized function.

    Parameters
    ----------
    data: ndarray or tuple of ndarray
        Can be either mat, omegac, omegar, or fun.
    ax:
        Matplotlib axis to plot it (created if `None` which is default).
    projection: None or str
        Which projection to use. `None` gives spherical coordinates.
    dpi : int or None (default)
        Resolution to use. Default (=None) is to use current Matplotlib figure settings.
    colorbar: bool
        Whether to add colorbar to plot.
    title: str or None
        Plot title.
    xlabel: str
        Label on x-axis.
    ylabel: str
        Label on y-axis.
    padding: int or None (default None)
        Amount (in pixels) of extra padding around image.
    N: int or None (default None)
        Up- or downsample to resolution N in plot.
    time: int or None (default None)
        Display time tag in plot.
    central_latitude: float (default 20)
        Latitude orientation (in degrees) for `projections='orthographic'`.
    central_longitude: float (default 30)
        Longitude orientation (in degrees) for `projections='orthographic'`.
    kwargs:
        Arguments to send to `ax.pcolormesh(...)`.

    Returns
    -------
        Object returned by `ax.pcolormesh(...)`.
    """
    use_cartopy = False

    # Convert and resample data if needed.
    if N is not None:
        data = resample(data, N)
    fun = as_fun(data)
    if np.iscomplexobj(fun):
        fun = fun.real

    # Create figure and axis if needed.
    if ax is None:
        if padding is None:
            if projection is None and title is None and colorbar is False:
                padding = 0
            else:
                padding = 2

        wpixels = fun.shape[1] + 2*padding
        hpixels = fun.shape[0] + 2*padding

        # Check projection type
        square_fig = False
        if _has_cartopy:
            if projection == "orthographic":
                projection = ccrs.Orthographic(central_latitude=central_latitude,
                                               central_longitude=central_longitude)
                wpixels = hpixels
                square_fig = True
            elif projection == "perspective":
                projection = ccrs.NearsidePerspective(central_latitude=central_latitude,
                                                      central_longitude=central_longitude)
                wpixels = hpixels
                square_fig = True
            if isinstance(projection, ccrs.CRS):
                # Cartopy object
                use_cartopy = True

        if dpi is None:
            default_title_height_pixels = round(25)
        else:
            default_title_height_pixels = round(25*dpi/100)
        if title is not None:
            hpixels += default_title_height_pixels
        default_color_bar_width_frac = 0.03
        default_color_bar_pad_frac = 0.02
        if colorbar:
            wpixels = round((1+default_color_bar_width_frac+default_color_bar_pad_frac)*wpixels)
            wpixels += 2*default_title_height_pixels

        # Create figure
        if dpi is None:
            figsize = plt.rcParams.get('figure.figsize')
            figsize = (figsize[0], figsize[0]*hpixels/wpixels)
        else:
            figsize = (wpixels/float(dpi), hpixels/float(dpi))
        fig = plt.figure(figsize=figsize)

        # Define axes
        left = padding/wpixels
        bottom = padding/hpixels
        height = fun.shape[0]/hpixels
        width = fun.shape[1]/wpixels if not square_fig else height

        # Create image axes
        ax = fig.add_axes([left, bottom, width, height], projection=projection)
        if title:
            ax.set_title(title)

        # Create colorbar axes
        if colorbar:
            cax = fig.add_axes([left + width + default_color_bar_pad_frac,
                                bottom, default_color_bar_width_frac, height])

    # Check limits if needed
    minmax = np.abs(fun).max()
    if 'vmin' not in kwargs:
        kwargs['vmin'] = -minmax
    if 'vmax' not in kwargs:
        kwargs['vmax'] = minmax

    # Check colormap
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'RdBu_r'

    # Create the plot
    lon = np.linspace(-np.pi, np.pi, fun.shape[1], endpoint=False)
    lat = np.linspace(-np.pi/2., np.pi/2., fun.shape[0])
    gridargs = {'color': 'black', 'alpha': 0.2}
    if use_cartopy:
        lon *= 360/(2*np.pi)
        lat *= 360/(2*np.pi)
        if 'transform' not in kwargs:
            kwargs['transform'] = ccrs.PlateCarree()
    # lon, lat = np.meshgrid(lon, lat)
    im = ax.pcolormesh(lon, lat, fun, rasterized=True, **kwargs)
    if gridon:
        if use_cartopy:
            ax.gridlines(draw_labels=False, dms=True, **gridargs)
        else:
            ax.grid(linestyle='-', **gridargs)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # add text with time-tag
    if time is not None:
        # place a text box in upper left in axes coords
        textstr = "time: {:.2f}".format(time)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top')
    if colorbar:
        im.figure.colorbar(mappable=im, cax=cax)

    return im


def create_animation2(filename, states, N=None, fps=25, preset='medium', extra_args=None,
                      codec='h264', title='QUFLOW animation',
                      progress_bar=True, progress_file=None, time=None, adaptive_scale=False, data2fun=as_fun, **kwargs):
    """
    Parameters
    ----------
    title
    filename
    states
    fps
    preset
    extra_args
    codec: str (default:'h264')
        ffmpeg codec. For accelerated Apple encoder, use 'h264_videotoolbox'
    progress_bar
    progress_file
    data2fun
    kwargs
        Sent to qf.plot(...)

    Returns
    -------

    """
    FFMpegWriter = anim.writers['ffmpeg']
    title = title.replace('QUFLOW', filename.replace('.mp4', ''))
    metadata = dict(title=title, artist='Matplotlib', comment='http://github.com/klasmodin/quflow')
    if extra_args is None:
        extra_args = []

    if preset == 'medium':
        if '-b:v' not in extra_args:
            extra_args += ['-b:v', '3000K']
        if '-preset' not in extra_args and codec == 'h264':
            extra_args += ['-preset', 'veryslow']
    elif preset == "low":
        if '-b:v' not in extra_args:
            extra_args += ['-b:v', '1500K']
    elif preset == "high":
        if '-preset' not in extra_args and codec == 'h264':
            extra_args += ['-preset', 'veryslow']
    elif preset == "twopass":
        if '-b:v' not in extra_args:
            extra_args += ['-b:v', '3000K']
        if '-preset' not in extra_args and codec == 'h264':
            extra_args += ['-preset', 'veryslow']


    # Create ffmpeg writer
    writer = FFMpegWriter(fps=fps, metadata=metadata, codec=codec, extra_args=extra_args)

    if N is not None:
        omega = resample(states[0], N)
    else:
        omega = states[0]
    f0 = data2fun(omega)

    with matplotlib.rc_context({'backend': 'Agg'}):

        if 'dpi' not in kwargs:
            kwargs['dpi'] = 100  # Default resolution

        im = plot2(f0, **kwargs)

        if time is not None:
            ax = im.axes
            textstr = "time: {}".format(time[0])
            timetag = ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top')
        
        if adaptive_scale:
            ax = im.axes
            textstr = "max: {}".format(0)
            maxtag = ax.text(0.02, 0.02, textstr, transform=ax.transAxes, verticalalignment='bottom')


        with writer.saving(im.figure, filename, dpi=100):

            if progress_bar and progress_file is None:
                from tqdm import trange
                stepiter = trange(states.shape[0], unit=' frames')
            elif progress_bar:
                from tqdm import trange
                stepiter = trange(states.shape[0], unit=' frames',
                                  file=progress_file, ascii=True, mininterval=10.0)
            else:
                stepiter = range(states.shape[0])
            for k in stepiter:
                if N is not None:
                    omega = resample(states[k], N)
                else:
                    omega = states[k]
                fun = data2fun(omega)
                # TODO: insert code here for saving img if state file is writable
                if hasattr(im, 'set_data'):
                    im.set_data(fun)
                elif hasattr(im, 'set_array'):
                    im.set_array(fun.ravel())
                else:
                    raise NotImplementedError("Could not find method for setting data.")
                if adaptive_scale:
                    minmax = np.abs(fun).max()
                    im.set_clim(vmin=-minmax, vmax=minmax)
                    maxtag.set_text("max: {:.2f}".format(minmax))
                if time is not None:
                    textstr = "time: {:.2f}".format(time[k])
                    timetag.set_text(textstr)
                writer.grab_frame()

        # Close figure (so it doesn't show up interactively)
        plt.close(fig=im.figure)

    if in_notebook():
        from IPython.display import Video
        return Video(filename, embed=False)


def spy(W, colorbar=True, logscale=True, ax=None):
    """
    Display the complex matrix elements using HSV colormap
    with H=arg(W), S=1, V=abs(W).

    Parameters
    ----------
    W: complex matrix
    colorbar: bool
    logscale: bool
    ax: Axes

    Returns
    -------
    fig: matplotlib imshow object
    """
    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1)
    Wabs = np.abs(W)
    if logscale:
        Wabs = np.log1p(Wabs)
    hsv_im = np.stack(((np.angle(W)%(2*np.pi))/(2*np.pi),
                        np.ones(W.shape),
                        Wabs/Wabs.max()), axis=-1)
    im = ax.imshow(hsv_to_rgb(hsv_im), cmap='hsv')
    if colorbar:
        cbar = ax.figure.colorbar(im, ticks=[0, 0.25, 0.5, 0.75, 1.0])
        cbar.ax.set_yticklabels(['1', 'i', '–1', '–i', '1'])  # vertically oriented colorbar

    return im
