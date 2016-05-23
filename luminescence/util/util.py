
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


def deconvolve(image, psf, iterations=7):
    '''
    performs the Richardson-Lucy deconvolution
    '''
    # return res.wiener(image, psf,  balance = 0.1, clip=True)
    return res.richardson_lucy(image, psf, iterations, clip=True)

def voltage_map(image, Ci, PL0=0, temp=300):
    '''
    returns a voltage map provide a calibration constant

        V = Vt x ln((PL - PL0)/Ci )
    '''

    Vt = const.k*temp/const.e
    return np.logspace((image - PL0) / Ci) * Vt

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


def show_edges(im, ax=None, sigma=1):

    edges2 = ft.canny(im, sigma=sigma)

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
