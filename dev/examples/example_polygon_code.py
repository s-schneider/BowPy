"""
Interactive tool to draw mask on an image or image-like array.
Adapted from matplotlib/examples/event_handling/poly_editor.py
"""
import numpy as np

# import matplotlib as mpl
# mpl.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.mlab import dist_point_to_segment
from matplotlib import path


class MaskCreator(object):
    """An interactive polygon editor.
    Parameters
    ----------
    poly_xy : list of (float, float)
        List of (x, y) coordinates used as vertices of the polygon.
    max_ds : float
        Max pixel distance to count as a vertex hit.
    Key-bindings
    ------------
    't' : toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them
    'd' : delete the vertex under point
    'i' : insert a vertex at point.  You must be within max_ds of the
          line connecting two existing vertices
    """
    def __init__(self, ax, poly_xy=None, max_ds=10):
        self.showverts = True
        self.max_ds = max_ds
        if poly_xy is None:
            poly_xy = default_vertices(ax)
        self.poly = Polygon(poly_xy, animated=True,
                            fc='y', ec='none', alpha=0.4)

        ax.add_patch(self.poly)
        ax.set_clip_on(False)
        ax.set_title("Click and drag a point to move it; "
                     "'i' to insert; 'd' to delete.\n"
                     "Close figure when done.")
        self.ax = ax

        x, y = zip(*self.poly.xy)
        self.line = plt.Line2D(x, y, color='none', marker='o', mfc='r',
                               alpha=0.2, animated=True)
        self._update_line()
        self.ax.add_line(self.line)

        self.poly.add_callback(self.poly_changed)
        self._ind = None # the active vert

        canvas = self.poly.figure.canvas
        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas

    def get_mask(self, shape):
        """Return image mask given by mask creator"""
        h, w = shape
        y, x = np.mgrid[:h, :w]
        points = np.transpose((x.ravel(), y.ravel()))
        mask = path.Path.contains_point(points)
        return mask.reshape(h, w)

    def poly_changed(self, poly):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        #Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        ignore = not self.showverts or event.inaxes is None or event.button != 1
        if ignore:
            return
        self._ind = self.get_ind_under_cursor(event)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        ignore = not self.showverts or event.button != 1
        if ignore:
            return
        self._ind = None

    def key_press_callback(self, event):
        'whenever a key is pressed'
        if not event.inaxes:
            return
        if event.key=='t':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        elif event.key=='d':
            ind = self.get_ind_under_cursor(event)
            if ind is None:
                return
            if ind == 0 or ind == self.last_vert_ind:
                print "Cannot delete root node"
                return
            self.poly.xy = [tup for i,tup in enumerate(self.poly.xy)
                                if i!=ind]
            self._update_line()
        elif event.key=='i':
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y # cursor coords
            for i in range(len(xys)-1):
                s0 = xys[i]
                s1 = xys[i+1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self.max_ds:
                    self.poly.xy = np.array(
                        list(self.poly.xy[:i+1]) +
                        [(event.xdata, event.ydata)] +
                        list(self.poly.xy[i+1:]))
                    self._update_line()
                    break
        self.canvas.draw()

    def motion_notify_callback(self, event):
        'on mouse movement'
        ignore = (not self.showverts or event.inaxes is None or
                  event.button != 1 or self._ind is None)
        if ignore:
            return
        x,y = event.xdata, event.ydata

        if self._ind == 0 or self._ind == self.last_vert_ind:
            self.poly.xy[0] = x,y
            self.poly.xy[self.last_vert_ind] = x,y
        else:
            self.poly.xy[self._ind] = x,y
        self._update_line()

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def _update_line(self):
        # save verts because polygon gets deleted when figure is closed
        self.verts = self.poly.xy
        self.last_vert_ind = len(self.poly.xy) - 1
        self.line.set_data(zip(*self.poly.xy))

    def get_ind_under_cursor(self, event):
        'get the index of the vertex under cursor if within max_ds tolerance'
        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x)**2 + (yt - event.y)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]
        if d[ind] >= self.max_ds:
            ind = None
        return ind


def default_vertices(ax):
    """Default to rectangle that has a quarter-width/height border."""
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    w = np.diff(xlims)
    h = np.diff(ylims)
    x1, x2 = xlims + w // 4 * np.array([1, -1])
    y1, y2 = ylims + h // 4 * np.array([1, -1])
    return ((x1, y1), (x1, y2), (x2, y2), (x2, y1))


def mask_creator_demo():
    img = np.random.uniform(0, 255, size=(100, 100))
    ax = plt.subplot(111)
    ax.imshow(img)

    mc = MaskCreator(ax)
    plt.show()

    mask = mc.get_mask(img.shape)
    img[~mask] = np.uint8(np.clip(img[~mask] - 100., 0, 255))
    plt.imshow(img)
    plt.title('Region outside of mask is darkened')
    plt.show()


if __name__ == '__main__':
    mask_creator_demo()
