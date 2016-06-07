
import matplotlib.pylab as plt


def _set_formatting(ax):
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def PlotImage_matplotlib(Image, ax=None, formatting=True):
    self.image_zval = Image
    self.numrows, self.numcols = Image.shape

    if formatting and ax is not None:
        _set_formatting()

    if ax is None:
        return plt.imshow(Image)
    else:
        return ax.imshow(Image)

def format_coord( x, y):

    col = int(x + 0.5)
    row = int(y + 0.5)
    if col >= 0 and col < self.numcols and row >= 0 and row < self.numrows:
        z = self.image_zval[row, col]
        # return 'x=%1.4f, y=%1.4f, z=%1.3e'%(x, y, z)
        return 'x=%1.4f, y=%1.4f, z=%1.2f' % (x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f' % (x, y)

