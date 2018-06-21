# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt

import skimage.io as io
import os
import scipy

from skimage.exposure import rescale_intensity
import skimage.transform as tf
import skimage.feature as ft

from PV_analysis.luminescence.imaging import core
import re

_imager_rot = {
    'Experimental': 90,
    'BTI': 0,
    None: 0,
}


def remove_non_ascii(fname):
    with open(fname) as fin:
        text = fin.readlines()

    text = re.sub(r'[^\x00-\x7F]+', ' ', ''.join(text))
    with open(fname, 'w') as fout:
        fout.write(text)

    return fname


def load_bti(fname, inf=None):
    '''
    Loads a luminescence image taken with BT output and returns
    the image class as a class.
    If the inf file is loaded, the images attributes are auto extracted.
    This extraction assumes the id of the file has not been changed.

    Current does not load the inf data

    inputs:
        fname: (str)
            The file path for the image file
        inf: (str, optional)
            The file path for the BTI's batch exported text file with
            information about the image aqusition. The image inf is loaded
            from this.
    returns:
        The luminesnce image class, with auto filled image aqusition data,
        and measurement data.
    '''

    # try and load the image
    try:
        os.path.isfile(fname)

        image = core.image(fname=fname.split(os.sep)[-1])
        image.image = loadimage_16bit(fname)
        image.fnames = os.path.basename(fname)

    except:
        print('File does not exist', fname)
        image = None

    # then try and extract the inf
    if inf is not None:
        try:
            data = np.genfromtxt(remove_non_ascii(
                inf), names=True, delimiter='\t')

            image.id = int(image.fnames.split('-')[0].strip(' '))

            # find it in the file
            index = data['id'] == image.id

            # things to extract from the BTI file, it only can import numbers
            settings = {
                'Jtm': 'Measured_Voltage_V',
                'Vtm': 'Measured_Current_A',
                'exposure': 'Exposure_Time_s',
                'photonflux': 'Photon_Flux_cm2_s1',
                'set_photonflux': 'Lightsource_Setpoint_Photon_Flux_cm2_s1',
                'thickness': 'Physical_Thickness_um',
                'doping': 'Electrical_Doping_Conc_cm3',
            }

            # grab the value in the file
            for key, value in settings.items():
                if value in data.dtype.names:
                    settings[key] = data[value][index][0]
                else:
                    settings[key] = None

            # get rid of the Nones
            settings = {k: v for k, v in settings.items() if v}

            image.attrs = settings

            # BTI doesn't do deconvolution so, it isn't
            image.deconvolved = False
        except:
            print('Something went wrong with the auto data extraction')

    return image


def loadimage_16bit(File, imager=None):
    '''
    Loads a 16 bit tiff the images as a numpy array.
    '''

    image = io.imread(os.path.join(File))
    image = rescale_intensity(np.abs(image), in_range=(0, 2.**16))

    image = tf.rotate(image, _imager_rot[imager])

    if np.amax(image) == 1:
        print('Warning: saturated pixel')

    return image


def saveimage_16bit(image,
                    fname='Test.tif',
                    folder=None,
                    rescale=True,
                    dtype=np.uint16,
                    imager=None):
    '''
    Saves an images as a 16 bit tiff
    '''

    # rotate the reverse direction
    image = tf.rotate(image, -1 * _imager_rot[imager])

    # if scaled to 0,1 then rescale back to 16 bit
    if rescale:
        # print 'rescaled'
        image = rescale_intensity(
            image, in_range=(0, 1), out_range=(0, 2**16))

    # Ensureing all the values are integers
    image = image.astype(dtype)

    folder = folder or ''

    image = io.imsave(
        os.path.join(folder, fname), image)
