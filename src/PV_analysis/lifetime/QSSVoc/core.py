
import numpy as np
import numbers
import scipy.constants as const
from PV_analysis.lifetime.core import lifetime as LTC


def Voc_2_deltan(V, doping, ni, temp):
    '''
    caculates the excess carrier density from a voltage
    inputs:
        V (array like):
            voltage in Volts
        Nd: (float in cm^-3)
            dopipng in
        ni: (float in (cm^-3))
            intrinsic carrier density
        temp: (float in K)
            Temperature
    outputs:
        nxc: The number of excess carriers.
    '''
    Vt = const.k * temp / const.e

    return (np.sqrt(doping**2. + 4. * ni**2 * (np.exp(V / Vt) - 1.)) - doping) / 2.


def dQscr(V, time, doping, esp=11.7, phi=1.1):
    '''
    This caculates the capactive effects of the space charge region. The space
    charge region can store chrage, which impacts a lifetime measurement.

    inputs:
        V: (array like, V)
            terminal voltage
        time: (array like, s)
            time stam for voltage measurements
        doping: (float)
            the bulk doping of the material
        esp: (float)
            the relative pemitivity of the material
        phi: (float)
            the electrostatic potential in equilibrium
    '''
    dvdt = np.gradient(V, time[2] - time[1])
    return np.sqrt(const.e * esp * const.epsilon_0 * doping / (
        2 * (phi - V))) * dvdt


class lifetime_Voc(LTC):
    '''

    this class can be used for

    a) self consisent QSSVoc analysis
    b) lifetime from suns Voc
    c) genralised Voc analysis.

    Inptuts:
        1. sample, doping
        2. thickness
        2. Generation. This is provided in gen_V and Fs
        3. time in seconts (time)
        4. Voltage (V)
    '''

    # raw measurements
    V = None
    gen_V = None

    Fs = None  # this is the generation calibration value

    _type = 'Voc'

    Qscr_correction = False

    def __init__(self, **kwargs):
        super(**kwargs).__init__()

    def _cal_nxc(self):

        if self.sample.nxc is None:
            self.sample.nxc = 0
        # get dn
        self.sample.nxc = Voc_2_deltan(
            self.V, self.sample.doping, self.sample.ni_eff, self.sample.temp)

    def cal_lifetime(self, analysis=None):

        self._cal_nxc()

        # get gen
        self.gen = self.gen_V * self.Fs
        self.gen = self._bg_correct(self.gen)
        # then do lifetime
        if self.Qscr_correction:
            other = dQscr(self.V, self.time, self.sample.doping)
        else:
            other = 0

        self._cal_lifetime(analysis=None, other=other)
