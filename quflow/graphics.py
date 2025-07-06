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


def plot(data, ax=None, projection='hammer', dpi=None, gridon=True, colorbar=False, title=None,
          xlabel="azimuth", ylabel="elevation", padding=None, N=None, time=None,
          central_latitude=20, central_longitude=30, gridargs=None, annotate=None, **kwargs):
    """
    Plot quantized functions on the sphere.

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
    annotate: callable(ax) or None (default None)
        Function to add annotations to the created axes `ax`. 
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
    default_gridargs = {'color': 'black', 'alpha': 0.2}
    if gridargs is not None:
        gridargs = {**default_gridargs, **gridargs}
    else:
        gridargs = default_gridargs

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

    # add user annotation
    if annotate is not None:
        ax.set_autoscale_on(False)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        annotate(ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    return im


plot2 = plot


def create_animation(filename, states, N=None, fps=25, preset='medium', extra_args=None,
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
        if '-b:v' not in extra_args:
            extra_args += ['-b:v', '8000K']
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

        im = plot(f0, **kwargs)

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
                    try:
                        minmax = np.max([minmaxold, np.abs(fun).max()])
                    except:
                        minmax = np.abs(fun).max()
                        minmaxold = minmax
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



create_animation2 = create_animation


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
