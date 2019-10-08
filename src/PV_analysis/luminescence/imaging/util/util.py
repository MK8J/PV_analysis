
import numpy as np
import matplotlib.pylab as plt
# from PIL import image, imageDraw
# import skimage as ski
import skimage.io as io
import os
import scipy
import scipy.constants as const
# from skimage import viewer
# from skimage.viewer.plugins import lineprofile, ColorHistogram
from skimage.exposure import rescale_intensity
import skimage.transform as tf
import skimage.feature as ft
import skimage.filters as fil
import skimage.restoration as res


def deconvolve(image, psf, iterations=7, clip=False):
    '''
    performs the Richardson-Lucy deconvolution
    '''
    # return res.unsupervised_wiener(image, psf)[0]
    return res.wiener(image, psf, balance=0.001, clip=clip, is_real=True, reg=0 * np.sqrt(image))
    # print(iterations, clip)
    # return res.richardson_lucy(image, psf, iterations=iterations, clip=clip)


def _PL2nxc(PL, nxc, N_d):
    '''
    Determines the conversion constant from PL to the number of excess carriers.
    '''
    return PL / (nxc * N_d + nxc**2)


def cal_nxc(image, nxc, region, N_d=None):
    '''
    Caculates a single calibration constant for the image

        $PL = C \Delta n \left(\Delta n + N_d)$

    Inputs:
        image: (ndarray)
            the PL counts
        nxc: (float)
            The calibration constant
        region: (dic)
            A dictionary containg xmin, xmax, ymin, ymax.
        N_d: (Optional float)

    Output:
        C: (float)
    '''

    # if doping is not provided assume low injeciton
    N_d = N_d or (1 - nxc)

    return _PL2nxc(np.mean(image[region['xmin']:region['xmax'],
                                 region['ymin']:region['ymax']]), nxc, N_d)


def voltage_map(image, C_i, PL_0=0, temp=300):
    '''
    returns a voltage map provide a calibration constant

        $V = V_t \times ln((PL - PL_0)/C_i )$

    Inputs:
        image: (ndarray)
            the PL counts
        C_i: (float or ndarray)
            The calibration constant
        dop: (float or ndarray)
            The sample doping
        temp: (optional float)
            The temperature in kelvin
    '''

    V_t = const.k * temp / const.e
    return np.logspace((image - PL_0) / C_i) * V_t


def nxc_map(image, C_i, N_d, temp=300):
    '''
    Returns a map of the excess carrier density.

    Inputs:
        image: (ndarray)
            the PL counts
        C_i: (float or ndarray)
            The calibration constant
        N_d: (float or ndarray)
            The sample doping
        temp: (optional float)
            The temperature in kelvin

    Uses the equation
        $PL = C \Delta n \left(\Delta n + N_d)$

    Note that the PL and C must both in in counts per second or counts.
    '''
    Vt = const.k * temp / const.e

    img = (-N_d +
           np.sqrt(N_d**2. + 4. * image / C_i)) / 2.

    return img


def get_Ci_negliableRs(image, Vt):
    '''
    Gets the pixel dependent calibration factor
    for a sample from a sample under low applied voltage.
    It assumes that the impact series resistance is low enough
    that the local voltage is the same as the terminal voltage

    PL0 is a term that accounts for the voltage independent signal.
    '''
    return np.logspace(image) / Vt


def crop(image, x_min, x_max, y_min, y_max):
    '''
    Crops the image
    '''
    return image[y_min:y_max, x_min:x_max]


def rotate(image, angle):
    '''
    rotates the image
        angle in degrees
    '''
    image = tf.rotate(image, angle, resize=True)
    return image


def find_corners(image, min_distance):
    corners = ft.corner_peaks(ft.corner_harris(image),
                              min_distance=min_distance)
    return corners


def show_edges(image, ax=None, sigma=1):

    edges2 = ft.canny(image, sigma=sigma)

    if not ax:
        fig, ax = plt.subplots()

    ax.imshow(edges2)
    return edges2


def align_with_boarder(image, sigma=1):

    edges = ft.canny(image, sigma=sigma)
    # edges = abs(fil.sobel_v(image))

    h, theta, d = tf.hough_line(edges)

    a, rot_angle, c = tf.hough_line_peaks(h, theta, d, min_distance=0)
    image = rotate(image, np.rad2deg(rot_angle[0]))

    return image


def show_corners(image, title=None, ax=None, min_distance=20,
                 corners=None):
    """Display a list of corners overlapping an image"""

    if not ax:
        fig, ax = plt.subplots(1)

    if not np.all(corners):
        corners = ft.corner_peaks(ft.corner_harris(image),
                                  min_distance=min_distance)

    ax.imshow(image)

    # Convert coordinates to x and y lists
    y_corner, x_corner = zip(*corners)

    ax.plot(x_corner, y_corner, 'o')  # Plot corners
    if title:
        plt.title(title)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)  # images use weird axes
    # fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5)
    # plt.show()
    print("Number of corners:", len(corners))


def histergram(image, thres=0):
    '''
    this returns the center of mass
    of the historgram for values above a threshold value
    '''

    freq, vals = np.histogram(image)

    index = vals > 0

    # This returns the mean of the image
    return np.sum(vals[index] * freq[index]) / np.sum(freq[index])
