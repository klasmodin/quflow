from .transforms import as_fun, mat2shr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import subplots
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.animation as anim
from matplotlib.colors import hsv_to_rgb
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.ndimage import map_coordinates

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
        if np.iscomplexobj(data) and data.shape[0] == data.shape[1]: # Assume mat
            omega = mat2shr(data)
        elif np.isrealobj(data) and 2*data.shape[0]-1 == data.shape[1]: # Assume fun
            if data.shape[0] == N:
                # We don't do anything
                return data
            X, Y = np.meshgrid(np.linspace(0, data.shape[0]-1, N, endpoint=True), 
                        np.linspace(0, data.shape[1], 2*N-1, endpoint=False), indexing='ij')
            fun_resampled = map_coordinates(data, np.array([X, Y]), order=1, mode='reflect')
            return fun_resampled
        else:
            raise NotImplementedError("Resampling this data is not supported yet.")
    elif data.ndim == 1:
        omega = data
    omega2 = np.zeros(N**2, dtype=omega.dtype)
    omega2[:min(N**2,omega.shape[0])] = omega[:min(N**2,omega.shape[0])]
    return omega2


def plot(data, fig=None, ax=None, projection='hammer', dpi=None, gridon=True, colorbar=False, title=None,
          xlabel="azimuth", ylabel="elevation", padding=None, N=None, time=None,
          central_latitude=20, central_longitude=30, gridargs=None, annotate=None, **kwargs):
    """
    Plot quantized functions on the sphere.

    Parameters
    ----------
    data: ndarray or tuple of ndarray
        Can be either mat, omegac, omegar, or fun.
    fig:
        Matplotlib figure to plot in (created if `None` which is default).
    ax:
        Matplotlib axis to plot in (created if `None` which is default).
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
    cax = None

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
        if fig is None:
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


def _get_ffmpeg_args(preset, extra_args, codec):
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
    
    return extra_args


class Animation(object):
    """
    Animation class for creating and managing video animations using Matplotlib and FFmpeg.
    This class provides a context manager interface for generating animations from quflow.QuSimulation data,
    handling frame updates, time tagging, and saving the output as a video file.

    filename : str
        The output filename for the animation video (e.g., 'animation.mp4').
        The initial image object to be animated. If None, a new plot will be created when updating with state data.
    fps : int, default 25
        Frames per second for the output video.
    N : int, optional
        Optional resampling parameter for the state data.
    preset : {'low', 'medium', 'high'}, default 'medium'
        FFmpeg preset for encoding speed/quality tradeoff.
    codec : str, default 'h264'
        Video codec to use for encoding.
    ffmpeg_args : str, optional
        Additional arguments to pass to FFmpeg.
    title : str, default "QUFLOW animation"
        Title metadata for the video file.
    **kwargs : optional
        Additional keyword arguments passed to the plotting function (e.g., dpi).
    
    Attributes
    ----------
    filename : str
        Output filename for the animation.
    N : int or None
        Resampling parameter.
    writer : matplotlib.animation.FFMpegWriter
        FFmpeg writer object for saving frames.
    im : matplotlib.image.AxesImage or similar
        Image object being animated.
    _plot_kwargs : dict
        Keyword arguments for the plotting function.
    figure : matplotlib.figure.Figure
        Figure object associated with the animation.
    canvas : matplotlib.backends.backend_agg.FigureCanvasAgg
        Canvas for rendering the figure.
    timetag : matplotlib.text.Text
        Text object for displaying the current time on the animation.

    Methods
    -------
    __enter__()
        Enter the context manager, returning self.
    __exit__(exc_type, exc_value, traceback)
        Exit the context manager, finishing the animation and closing the figure.
    setup()
        Set up the animation writer and figure canvas.
    finish()
        Finalize the animation, close the figure, and optionally display the video in a notebook.
    update(state=None, time=None, grab=True, im=None)
        Update the animation frame with new state data and/or time tag, and optionally grab the frame for the video.
    
    Notes
    -----
    - The class is designed to be used as a context manager for safe resource handling.
    - The `update` method can be called repeatedly to add frames to the animation.
    - Time tags are automatically added or updated on the animation frames.
    """

    def __init__(self, 
                 filename: str, 
                 im = None,
                 fps: int = 25,
                 N = None,
                 preset: str = 'medium',
                 codec: str = 'h264',
                 ffmpeg_args: str = None,
                 title: str = "QUFLOW animation",
                 **kwargs
                 ):
        
        FFMpegWriter = anim.writers['ffmpeg']
        title = title.replace('QUFLOW', filename.replace('.mp4', ''))
        # metadata = dict(title=title, artist='Matplotlib', comment='http://github.com/klasmodin/quflow')
        metadata = dict(artist='Quflow/Matplotlib', comment='http://github.com/klasmodin/quflow')
        extra_args = _get_ffmpeg_args(preset, ffmpeg_args, codec)

        # File
        self.filename = filename

        # Resampling N
        self.N = N

        # Create ffmpeg writer
        self.writer = FFMpegWriter(fps=fps, metadata=metadata, codec=codec, extra_args=extra_args)

        # Plot image
        self.im = im

        # Save quflow.plot arguments
        if 'dpi' not in kwargs:
            kwargs['dpi'] = 100  # Default resolution
        self._plot_kwargs = kwargs

        # Setup writer if possible
        if self.im is not None:
            self.setup()


    def __enter__(self):
        # Enter context
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        # Exit context
        self.finish()
        # Return False to propagate the exception, True to suppress it
        return False


    def setup(self):
        self.figure = self.im.figure
        self.canvas = FigureCanvasAgg(self.figure)
        self.writer.setup(self.figure, self.filename, dpi=100)


    def finish(self):
        self.writer.finish()
        # Close figure (so it doesn't show up interactively)
        plt.close(fig=self.figure)

        # if in_notebook():
        #     from IPython.display import Video
        #     return Video(self.filename, embed=False)


    def update(self, 
               state: np.ndarray = None, 
               time = None,
               grab = True,
               im = None):
        """
        Updates the animation frame with the given state and time, and optionally grabs the frame for the video.

        Parameters
        ----------
        state : np.ndarray, optional
            The new state data to be visualized. If provided, updates the image data accordingly.
        time : float or str, optional
            The current time to display as a tag on the animation. Can be a float or a string.
        grab : bool, default True
            Whether to grab the current frame and add it to the animation.
        im : matplotlib.image.AxesImage or similar, optional
            The image object to update. If not provided, uses the default image associated with the animation.
        Raises
        ------
        AttributeError
            If the image object does not have a method for setting data.
        Notes
        -----
        - If `state` is provided and no image exists, a new plot is created.
        - If `time` is provided, a time tag is added or updated on the animation.
        - If `grab` is True, the current frame is captured for the animation.
        """
        
        if im is None:
            im = self.im

        if state is not None:
            if im is None:
                # Create default plot
                fun = as_fun(state)
                if self.N is not None:
                    fun = resample(fun, self.N)
                self.im = plot(fun, **self._plot_kwargs)
                im = self.im
                self.setup()
            else:
                fun = as_fun(state)
                if self.N is not None:
                    fun = resample(fun, self.N)
                if hasattr(im, 'set_data'):
                    im.set_data(fun)
                elif hasattr(im, 'set_array'):
                    im.set_array(fun.ravel())
                else:
                    raise AttributeError("Could not find method for setting data.")
        
        # Add time
        if time is not None:
            if isinstance(time, str):
                textstr = time
            elif abs(time) < 100:
                textstr = "t={:.1f}".format(time)
            else:
                textstr = "t={:.0f}".format(time)
            # textstr = time if isinstance(time, str) else "t = {:.1f}".format(time)
            if not hasattr(self, 'timetag'):
                ax = im.axes
                bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
                width, height = bbox.width, bbox.height
                # fig_width, fig_height = self.figure.get_size_inches()
                scale_factor = width / 8  # Assuming 8 is the base size
                self.timetag = ax.text(0.01, 0.91, textstr, 
                                       transform=ax.transAxes, 
                                       verticalalignment='baseline',
                                       fontsize=24*scale_factor)
            else:
                self.timetag.set_text(textstr)

        # Grab frame from figure
        if grab:
            self.writer.grab_frame()



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
    extra_args = _get_ffmpeg_args(preset, extra_args, codec)

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
