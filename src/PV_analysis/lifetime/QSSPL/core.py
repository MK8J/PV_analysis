

import numpy as np
import scipy.constants as const
from PV_analysis.lifetime.core import lifetime as LTC
from semiconductor.recombination.intrinsic import Radiative as rad


def PL_2_deltan(PL, doping, Ai, temp):
    '''
    caculates the excess carrier density from a pl intenisty
    :
        PL (array like):
            voltage in Volts
        doping: (float in cm^-3)
            The ionised number of dopants
        Ai: (float in (cm^-3))
            proportionaity factor
        temp: (float in K)
            Temperature
    outputs:
        nxc: The number of excess carriers.
    '''
    nxc = np.ones(PL.shape[0]) * 1e10
    i = 1

    # loop through to update B
    for i in range(10):

        maj_car_den = doping + nxc

        B = rad(nxc=nxc, Nd=doping, temp=temp)._get_B(
        )

        _temp = (-maj_car_den +
                 np.sqrt(np.absolute(
                     (maj_car_den)**2 + 4 * PL * Ai / B))) / 2

        i = np.average(np.absolute(temp - nxc) / nxc)
        nxc = _temp

    return nxc


class lifetime_PL(LTC):

    # measurement settings
    _m_settings = None

    # raw measurements
    I_PL = None
    gen_V = None

    Fs = None  # this is the generation calibration value
    Ai = None

    _type = 'PL'

    def __init__(self, **kwargs):
        super(**kwargs).__init__()

    def cal_lifetime(self, analysis=None):

        # background correct the data
        self.I_PL = self._bg_correct(self.I_PL)
        self.gen = self._bg_correct(self.gen)

        # get dn
        self.nxc = PL_2_deltan(
            self.I_PL, self.sample.doping, self.Ai, self.temp)

        # get gen
        self.gen = self.gen_V * self.Fs / self.sample.thickness * \
            self.sample.optical_c
        # then do lifetime
        self._cal_lifetime(analysis=None)
