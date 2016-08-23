
import numpy as np
import scipy.constants as const
from semiconductor.electrical.mobility import Mobility
from PV_analysis.lifetime.core import lifetime as LTC


def voltage2conductance(voltage, coil_constants, method='quad'):
    '''
    converts a voltage from a coil into a measured condutances

    inputs:
        voltage: (np array)
            The coil output voltage
        coil_constants:  (dic)
            A dictionary containing the coil constants a, b and c.
        method: (str: quad or sinton)
            A string determining how to cacualte the conductance.
    '''

    convertion = {
        'sinton': _voltage2conductance_sinton,
        'quad': _voltage2conductance_quadratic
    }

    return convertion[method](voltage, coil_constants)


def _voltage2conductance_sinton(V, coil_constants):
    '''
    assumes the constants provided follow

    conductance = (V-coil_constants[c])**2 coil_constants[a] + (V-coil_constants[c]) coil_constants[b]
    '''

    return (V - coil_constants['c'])**2 * coil_constants['a'] + (V - coil_constants['c']) * coil_constants['b']


def _voltage2conductance_quadratic(V, coil_constants):
    '''
    assumes the constants provided follow

    conductance = V**2 coil_constants[a] + V coil_constants[b] + coil_constants[c]
    '''

    return V**2 * coil_constants['a'] + V * coil_constants['b'] + coil_constants['c']


def conductance2deltan(cond, Na, Nd, thickness, mob_author=None, temp=300):
    '''
    returns the excess conductance for a sample
    '''

    nxc = 1e10 * np.ones(cond.shape[0])
    i = 1
    while (i > 0.001):

        _temp = cond / const.e / Mobility(
        ).mobility_sum(nxc=nxc, Na=Na,
                       Nd=Nd, temp=temp, author=mob_author
                       ) / thickness

        _temp[_temp < 0.] = 0.
        index = np.nonzero(nxc)
        i = np.average(abs(_temp[index] - nxc[index]) / nxc[index])

        nxc = _temp

    return nxc


class lifetime_QSSPC(LTC):

    # raw measurements
    PC = None
    gen_V = None

    Fs = None  # this is the generation calibration value
    coil_constants = {'a': None,
                      'b': None,
                      'c': None}

    calibration_method = 'quad'  # options are sinton and quad
    dark_conductance = None

    _type = 'PC'

    def __init__(self, **kwargs):
        super(**kwargs).__init__()
        self.analysis_options['mobility'] = None
        self.analysis_options['ni'] = None
        self.analysis_options['Bg_narrowing'] = None

    def _cal_nxc(self, dark_conductance=None):
        self.nxc = voltage2conductance(
            voltage=self.PC, coil_constants=self.coil_constants,
            method=self.calibration_method)

        # background correct the data, generation and conductance
        if dark_conductance:
            self.nxc -= voltage2conductance(
                voltage=dark_conductance, coil_constants=self.coil_constants,
                method=self.calibration_method)

        else:
            self.nxc = self._bg_correct(self.nxc)

        self.gen_V = self._bg_correct(self.gen_V)

        # get nxc
        self.nxc = conductance2deltan(
            cond=self.nxc, Na=self.sample.Na,
            Nd=self.sample.Nd, thickness=self.sample.thickness,
            mob_author=self.analysis_options['mobility'],
            temp=self.sample.temp)

    def cal_lifetime(self, analysis=None, dark_conductance=None):
        # get conductance
        self._cal_mcd(dark_conductance)

        # get gen
        self.gen = self.gen_V * self.Fs

        # then do lifetime
        self._cal_lifetime(analysis=None)

    @property
    def mobility_sum(self):
        if isinstance(self.analysis_options['mobility_sum'], numbers.Number):
            _mu = self.analysis_options['mobility_sum']
        elif isinstance(self.analysis_options['mobility_sum'], np.ndarray):
            _mu = self.analysis_options['mobility_sum']
        else:
            _mu = Mobility(
                author=self.analysis_options['mobility'],
                material='Si', temp=self.sample.temp,
                nxc=self.nxc, Na=self.sample.Na, Nd=self.sample.Nd
            ).mobility_sum()

        return _mu

    @mobility_sum.setter
    def mobility_sum(self, value):
        self.analysis_options['mobility_sum'] = val
