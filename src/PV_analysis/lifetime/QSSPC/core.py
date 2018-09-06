import numpy as np
import numbers
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

    conductance = (V-coil_constants[c])**2 coil_constants[a] +
     (V-coil_constants[c]) coil_constants[b]
    '''
    V0 = V - coil_constants['c']
    return (V0)**2 * coil_constants['a'] + (V0) * coil_constants['b']


def _voltage2conductance_quadratic(V, coil_constants):
    '''
    assumes the constants provided follow

    conductance = V**2 coil_constants[a] + V coil_constants[b] +
     coil_constants[c]

    inputs:
        V: (array like, in volts)
            measured voltage
        coil_constants: (dic)
            coil constants, named a, b, and c as see in the equation above

    '''

    return V**2 * coil_constants['a'] + V * coil_constants['b'] +\
        coil_constants['c']


def conductance2deltan(conductance, Na, Nd, thickness, mobility_sum, temp):
    '''
    returns the excess conductance for a sample

    inputs:

        conductance: (array like, semens)
            The photocondutance. It should be corrected by the dark condutances
        Na: (float, cm^-3)
            The acceptor density.
        Nd: (float, cm^-3)
            The donar density.
        thickness: (float, cm)
            The thickness of the device
        mobility_sum: (array like or string or None)
            The mobility sum. This can either be values, a string corresponding to an mobility
            author in the semiconductor module, or None. If not it uses the default model
            from the semiconductor module.
        temp: (float, kelvin)
            The temperature

    '''
    if isinstance(mobility_sum, np.ndarray):
        nxc = _conductance2deltan_array(conductance, thickness, mobility_sum)

    elif type(mobility_sum) == str:
        nxc = _conductance2deltan_model(
            conductance, Na, Nd, thickness, mobility_sum, temp)
    elif mobility_sum is None:
        nxc = _conductance2deltan_model(
            conductance, Na, Nd, thickness, mobility_sum, temp)

    return nxc


def _conductance2deltan_model(conductance, Na, Nd, thickness,
                              mobility_sum_author, temp):
    '''
    returns the excess conductance for a sample
    '''
    assert type(mobility_sum_author) == str or mobility_sum_author is None

    nxc = 1e10 * np.ones(conductance.shape[0])
    i = 1

    while (i > 0.0001):
        mobility_sum = Mobility(
        ).mobility_sum(nxc=nxc, Na=Na,
                       Nd=Nd, temp=temp, author=mobility_sum_author
                       )
        _temp = _conductance2deltan_array(conductance, thickness, mobility_sum)

        # make sure we are not loosing carriers
        _temp[_temp < 0.] = 0.
        # ignore points where there are no carriers
        index = np.nonzero(nxc)
        i = np.average(abs(_temp[index] - nxc[index]) / nxc[index])

        nxc = _temp
    # return the value
    return nxc


def _conductance2deltan_array(conductance, thickness, mobility_sum):
    '''
    returns the excess conductance for a sample for a given conductance,
    thickness and mobility sum
    '''

    return conductance / const.e / mobility_sum / thickness


class lifetime_QSSPC(LTC):

    # raw measurements
    PC = None
    gen_V = None

    Fs = None  # this is the generation calibration value
    coil_constants = {'a': None,
                      'b': None,
                      'c': None}

    calibration_method = 'quad'  # options are sinton and quad
    dark_voltage = None

    _type = 'PC'

    def __init__(self, **kwargs):
        super(**kwargs).__init__()
        self.analysis_options['mobility'] = None
        self.analysis_options['mobility_sum'] = None

    def _cal_nxc(self, dark_voltage=None):
        '''
        Caculates the excess carrier density from the conductance
        using PC coil cosntants, and a mobility model.
        '''
        # this assumes the voltage prodided is the meaured voltage,
        # and does not have a value background corrected
        self.sample.nxc = voltage2conductance(
            voltage=self.PC, coil_constants=self.coil_constants,
            method=self.calibration_method)

        # background correct the data, generation and conductance
        if dark_voltage:
            self.sample.nxc -= voltage2conductance(
                voltage=dark_voltage, coil_constants=self.coil_constants,
                method=self.calibration_method)

        else:
            # this does not do anyhting
            print('here', self.sample.nxc[-1])
            self.sample.nxc = self._bg_correct(self.sample.nxc)

            print('here', self.sample.nxc[-1])
        self.gen_V = self._bg_correct(self.gen_V)

        # get nxc
        self.sample.nxc = conductance2deltan(
            conductance=self.sample.nxc, Na=self.sample.Na,
            Nd=self.sample.Nd, thickness=self.sample.thickness,
            mobility_sum=self.mobility_sum,
            temp=self.sample.temp)

    def cal_lifetime(self, analysis=None, dark_voltage=None):
        '''
        Calculates the lifetime from the measure voltages.

        inputs:
            analysis: (str, optional)
                the method of analysis. Options include: generalised,        transient, and steadystate
            dark_voltage: (float, optional)
                Provides the opturnity to updated the assigned dark_voltage value. If dark_voltage was never assigned the PC data is background subtracted.

        '''
        # get conductance
        # use the provided dark voltage, or the already assigned
        self.dark_voltage = dark_voltage or self.dark_voltage
        self._cal_nxc(self.dark_voltage)

        # get gen
        self.gen = self.gen_V * self.Fs

        # then do lifetime
        self._cal_lifetime(analysis=analysis)

    @property
    def mobility_sum(self):
        if isinstance(self.analysis_options['mobility_sum'], numbers.Number):
            _mu_sum = self.analysis_options['mobility_sum']
        elif isinstance(self.analysis_options['mobility_sum'], np.ndarray):
            _mu_sum = self.analysis_options['mobility_sum']
        else:
            _mu_sum = self.analysis_options['mobility']

        return _mu_sum

    @mobility_sum.setter
    def mobility_sum(self, value):
        self.analysis_options['mobility_sum'] = value


if __name__ == '__main__':
    print('ok')
    print(_conductance2deltan_model(np.asarray([8.640E-04, 7.477E-03]),
                                    Na=np.asarray([9.14E+15]), Nd=np.asarray([0]),
                                    thickness=0.018, mobility_sum_author='Dannhauser-Krausse_1972', temp=300))
