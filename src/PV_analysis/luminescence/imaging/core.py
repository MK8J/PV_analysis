
from PV_analysis.common import IO, sample


def FF_theoritical(image, d_adj=1):
    '''
    Calculates and corrects for flat field effects, assuming a cos^2 dependance
    This assumes a square CCD
    '''
    # get the x coords
    x = np.linspace(1, image.shape[0],
                    image.shape[0]) - (image.shape[0] + 1) / 2
    # the y coords
    y = x

    # make a mesh
    for i in range(image.shape[0] - 1):
        x = np.vstack([x, y])

    y = x.T

    # cal r
    r = np.sqrt((x)**2 + (y)**2)

    def cos2(f_adj=1):
        '''
        A cos squared function

        inputs:
            f_adj: (float, optinal, default =1)
                an adjument to the frequency. A value of 1 sets the cos of frequency corresponding to a minimum
                at the corners of the image, and max in the cener. This value scales the frequency away from this value.

        output:
            An adjusted image by the flat field

        '''
        # calcualte the period
        T = np.amax(r) * 2 / np.pi
        # calcualte the frequency
        f = 1. / T * f_adj
        # get the cos and normalised max to 1.
        w = np.cos(f * (r))**2
        return w / np.amax(w)

    return image / cos2(f_adj)


class image():
    '''
    A class to handel PL images
    '''

    image = None
    fname = None
    # measurement settings
    exposure = None
    photonflux = None

    # electrical settings
    JL = None
    Vtm = None
    Jtm = None
    # file names
    fnames = None

    # deconvolution status
    deconvolved = None
    _warnings = False

    def __init__(self, **kwargs):

        self.sample = sample()
        self.attrs = kwargs

    @property
    def image_norm(self):
        assert self.image is not None
        assert self.exposure is not None
        return self.image / self.exposure

    @property
    def attrs(self):
        return {None}

    @attrs.setter
    def attrs(self, kwargs):
        self.other_inf = {}
        assert type(kwargs) == dict

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.sample, key):
                setattr(self.sample, key, value)
            else:
                if self._warnings:
                    print('Attribute {0} not found'.format(key))
                    print('Attribute set in dic other_inf')
                self.other_inf[key] = value
