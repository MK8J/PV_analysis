

import numpy as np
import scipy.constants as const
from PV_analysis.lifetime.core import lifetime as LTC
from semiconductor.recombination import Radiative


def PL_2_deltan(PL, Na, Nd, Ai, temp):
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
    doping = abs(Na - Nd)
    for i in range(10):

        maj_car_den = doping

        B = Radiative(material='Si', temp=temp, Na=Na, Nd=Nd).get_B(nxc)

        _temp = (-maj_car_den +
                 np.sqrt(np.absolute(
                     (maj_car_den)**2 + 4. * PL * Ai / B))) / 2.

        # i = np.average(np.absolute(_temp - nxc) / nxc)
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

    gain_pl = None
    gain_gen = None

    def __init__(self, **kwargs):
        super(**kwargs).__init__()

    def cal_lifetime(self, analysis=None):

        # background correct the data
        self.I_PL = self._bg_correct(self.I_PL)
        self.gen_V = self._bg_correct(self.gen_V)  # * gain_gen

        # get dn
        self.sample.nxc = PL_2_deltan(
            PL=self.I_PL, Na=self.sample.Na, Nd=self.sample.Nd, Ai=self.Ai * gain_pl,
            temp=self.sample.temp)

        # get gen
        self.gen = self.gen_V * self.Fs
        # then do lifetime
        self._cal_lifetime(analysis=None)
