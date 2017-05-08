from __future__ import absolute_import
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook
from matplotlib.lines import Line2D
from matplotlib import path as mplPath
import numpy as np
import scipy.spatial as spatial

def fmt(x, y):
    return 'x: {x:0.2f}\ny: {y:0.2f}'.format(x = x, y = y)

class DataCursor(object):
    # http://stackoverflow.com/a/4674445/190597
    """A simple data cursor widget that displays the x,y location of a
    matplotlib artist when it is selected.
    Example:

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    scat = ax.plot(x, y)
    DataCursor(scat, x, y)
    plt.show()


    """
    def __init__(self, artists, x = [], y = [], tolerance = 5, offsets = (-20, 20),
                 formatter = fmt, display_all = False):
        """Create the data cursor and connect it to the relevant figure.
        "artists" is the matplotlib artist or sequence of artists that will be 
            selected. 
        "tolerance" is the radius (in points) that the mouse click must be
            within to select the artist.
        "offsets" is a tuple of (x,y) offsets in points from the selected
            point to the displayed annotation box
        "formatter" is a callback function which takes 2 numeric arguments and
            returns a string
        "display_all" controls whether more than one annotation box will
            be shown if there are multiple axes.  Only one will be shown
            per-axis, regardless. 
        """
        self._points = np.column_stack((x,y))
        self.formatter = formatter
        self.offsets = offsets
        self.display_all = display_all
        if not cbook.iterable(artists):
            artists = [artists]
        self.artists = artists
        self.axes = tuple(set(art.axes for art in self.artists))
        self.figures = tuple(set(ax.figure for ax in self.axes))

        self.annotations = {}
        for ax in self.axes:
            self.annotations[ax] = self.annotate(ax)

        for artist in self.artists:
            artist.set_picker(tolerance)
        for fig in self.figures:
            fig.canvas.mpl_connect('pick_event', self)

    def annotate(self, ax):
        """Draws and hides the annotation box for the given axis "ax"."""
        annotation = ax.annotate(self.formatter, xy = (0, 0), ha = 'right',
                xytext = self.offsets, textcoords = 'offset points', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
                )
        annotation.set_visible(False)
        return annotation

    def snap(self, x, y):
        """Return the value in self._points closest to (x, y).
        """
        idx = np.nanargmin(((self._points - (x,y))**2).sum(axis = -1))
        return self._points[idx]
    def __call__(self, event):
        """Intended to be called through "mpl_connect"."""
        # Rather than trying to interpolate, just display the clicked coords
        # This will only be called if it's within "tolerance", anyway.
        x, y = event.mouseevent.xdata, event.mouseevent.ydata
        annotation = self.annotations[event.artist.axes]
        if x is not None:
            if not self.display_all:
                # Hide any other annotation boxes...
                for ann in self.annotations.values():
                    ann.set_visible(False)
            # Update the annotation in the current axis..
            x, y = self.snap(x, y)
            annotation.xy = x, y
            annotation.set_text(self.formatter(x, y))
            annotation.set_visible(True)
            event.canvas.draw()
            

class FollowDotCursor(object):
    """
    Display the x,y location of the nearest data point.

    Example:

    x=[1,2,3,4,5]
    y=[6,7,8,9,10]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y)
    cursor = FollowDotCursor(ax, x, y)
    plt.show()

    """
    def __init__(self, ax, x, y, tolerance=5, formatter=fmt, offsets=(-20, 20)):
        try:
            x = np.asarray(x, dtype='float')
        except (TypeError, ValueError):
            x = np.asarray(mdates.date2num(x), dtype='float')
        y = np.asarray(y, dtype='float')
        self._points = np.column_stack((x, y))
        self.offsets = offsets
        self.scale = x.ptp()
        self.scale = y.ptp() / self.scale if self.scale else 1
        self.tree = spatial.cKDTree(self.scaled(self._points))
        self.formatter = formatter
        self.tolerance = tolerance
        self.ax = ax
        self.fig = ax.figure
        self.ax.xaxis.set_label_position('top')
        self.dot = ax.scatter(
            [x.min()], [y.min()], s=130, color='green', alpha=0.7)
        self.annotation = self.setup_annotation()
        plt.connect('motion_notify_event', self)

    def scaled(self, points):
        points = np.asarray(points)
        return points * (self.scale, 1)

    def __call__(self, event):
        ax = self.ax
        # event.inaxes is always the current axis. If you use twinx, ax could be
        # a different axis.
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
        elif event.inaxes is None:
            return
        else:
            inv = ax.transData.inverted()
            x, y = inv.transform([(event.x, event.y)]).ravel()
        annotation = self.annotation
        x, y = self.snap(x, y)
        annotation.xy = x, y
        annotation.set_text(self.formatter(x, y))
        self.dot.set_offsets((x, y))
        bbox = ax.viewLim
        event.canvas.draw()

    def setup_annotation(self):
        """Draw and hide the annotation box."""
        annotation = self.ax.annotate(
            '', xy=(0, 0), ha = 'right',
            xytext = self.offsets, textcoords = 'offset points', va = 'bottom',
            bbox = dict(
                boxstyle='round,pad=0.5', fc='yellow', alpha=0.75),
            arrowprops = dict(
                arrowstyle='->', connectionstyle='arc3,rad=0'))
        return annotation

    def snap(self, x, y):
        """Return the value in self.tree closest to x, y."""
        dist, idx = self.tree.query(self.scaled((x, y)), k=1, p=1)
        try:
            return self._points[idx]
        except IndexError:
            # IndexError: index out of bounds
            return self._points[0]

def get_polygon(data, no_of_vert=4, xlabel=None, xticks=None, ylabel=None, yticks=None, fs=25):
    """
    Interactive function to pick a polygon out of a figure and receive the vertices of it.
    :param data:
    :type:
    
    :param no_of_vert: number of vertices, default 4, 
    :type no_of_vert: int
    """
    from bowpy.util.polygon_interactor import PolygonInteractor
    from matplotlib.patches import Polygon
    
    no_of_vert = int(no_of_vert)
    # Define shape of polygon.
    try:
        x, y = xticks.max(), yticks.max() 
        xmin= -x/10.
        xmax= x/10.
        ymin= y - 3.*y/2.
        ymax= y - 3.*y/4.

    except AttributeError:
        y,x = data.shape
        xmin= -x/10.
        xmax= x/10.
        ymin= y - 3.*y/2.
        ymax= y - 3.*y/4.
        
    xs = []
    for i in range(no_of_vert):
        if i >= no_of_vert/2:
            xs.append(xmax)
        else:
            xs.append(xmin)

    ys = np.linspace(ymin, ymax, no_of_vert/2)
    ys = np.append(ys,ys[::-1]).tolist()

    poly = Polygon(list(zip(xs, ys)), animated=True, closed=False, fill=False)
    
    # Add polygon to figure.
    fig, ax = plt.subplots()
    ax.add_patch(poly)
    p = PolygonInteractor(ax, poly)
    plt.title("Pick polygon, close figure to save vertices")
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel(ylabel, fontsize=fs)

    try:
        im = ax.imshow(abs(data), aspect='auto', extent=(xticks.min(), xticks.max(), 0, yticks.max()), interpolation='none')
    except AttributeError:
        im = ax.imshow(abs(data), aspect='auto', interpolation='none')

    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.set_ylabel('R', fontsize=fs)
    mpl.rcParams['xtick.labelsize'] = fs
    mpl.rcParams['ytick.labelsize'] = fs
    ax.tick_params(axis='both', which='major', labelsize=fs)


    plt.show()      
    print("Calculate area inside chosen polygon\n")
    try:
        vertices = (poly.get_path().vertices)
        vert_tmp = []
        xticks.sort()
        yticks.sort()
        for fkvertex in vertices:
            vert_tmp.append([np.abs(xticks-fkvertex[0]).argmin(), np.abs(yticks[::-1]-fkvertex[1]).argmin()])
        vertices = np.array(vert_tmp)   
        
    except AttributeError:
        vertices = (poly.get_path().vertices).astype('int')

    indicies = convert_polygon_to_flat_index(data, vertices)
    return indicies


def convert_polygon_to_flat_index(data, vertices):
    """
    Converts points insde of a polygon defined by its vertices, taken of an imshow plot of data,to 
    flat-indicies. Does NOT include the border of the polygon.
    
    :param data: speaks for itself
    :type data: numpy.ndarray

    :param vertices: also...
    :type vertices: numpy.ndarray
    
    """

    # check if points are inside polygon. Be careful with the indicies, np and mpl
    # handle them exactly opposed.
    polygon = mplPath.Path(vertices)
    arr = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if polygon.contains_point([j,i]):
                arr.append([j,i])
    arr = map(list, zip(*arr))

    flat_index= np.ravel_multi_index(arr, data.conj().transpose().shape).astype('int').tolist()

    return(flat_index)  

def pick_data(x, y, xlabel, ylabel, title):
        
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_title(title)
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)
    line, = ax1.plot(x, y , picker=5)  # 5 points tolerance
    
    global PickByHand
    PickByHand = []
    def onpick1(event):
        if isinstance(event.artist, Line2D):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            pick = zip(np.take(xdata, ind), np.take(ydata, ind))
            print('onpick1 line:', zip(np.take(xdata, ind), np.take(ydata, ind)) )
            PickByHand.append(pick)

    fig.canvas.mpl_connect('pick_event', onpick1)
    plt.show()
    return PickByHand

