<<<<<<< Updated upstream

import numpy as np
import scipy.constants as const
from semiconductor.electrical.mobility import Mobility


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

    convertion ={
    'sinton': _voltage2conductance_sinton,
    'quad': _voltage2conductance_quadratic
    }

    return convertion[method](voltage, coil_constants)

def _voltage2conductance_sinton(V, coil_constants):
    '''
    assumes the constants provided follow

    conductance = (V-coil_constants[c])**2 coil_constants[a] + (V-coil_constants[c]) coil_constants[b]
    '''


    return (V-coil_constants['c'])**2*coil_constants['a'] +(V-coil_constants['c'])*coil_constants['b']

def _voltage2conductance_quadratic(V, coil_constants):
    '''
    assumes the constants provided follow

    conductance = V**2 coil_constants[a] + V coil_constants[b] + coil_constants[c]
    '''


    return V**2*coil_constants['a'] +V*coil_constants['b'] + coil_constants['c']

def conductance2deltan(cond, Na, Nd, thickness):
    '''
    returns the excess conductance for a sample
    '''

    nxc = 1e10 * np.ones(cond.shape[0])
    i=1
    while (i > 0.001):

        temp = cond/const.e/Mobility().mobility_sum(nxc=nxc, Na=Na, Nd=Nd, temp=300)/thickness
        i = np.average(abs(temp-nxc)/nxc)

        nxc = temp

    return nxc
=======

import numpy as np
import scipy.constants as const
from semiconductor.electrical.mobility import Mobility


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

    convertion ={
    'sinton': _voltage2conductance_sinton,
    'quad': _voltage2conductance_quadratic
    }

    return convertion[method](voltage, coil_constants)

def _voltage2conductance_sinton(V, coil_constants):
    '''
    assumes the constants provided follow

    conductance = (V-coil_constants[c])**2 coil_constants[a] + (V-coil_constants[c]) coil_constants[b]
    '''


    return (V-coil_constants['c'])**2*coil_constants['a'] +(V-coil_constants['c'])*coil_constants['b']

def _voltage2conductance_quadratic(V, coil_constants):
    '''
    assumes the constants provided follow

    conductance = V**2 coil_constants[a] + V coil_constants[b] + coil_constants[c]
    '''


    return V**2*coil_constants['a'] +V*coil_constants['b'] + coil_constants['c']

def conductance2deltan(cond, Na, Nd, thickness):
    '''
    returns the excess conductance for a sample
    '''

    nxc = 1e10 * np.ones(cond.shape[0])
    i=1
    while (i > 0.001):

        temp = cond/const.e/Mobility().mobility_sum(nxc=nxc, Na=Na, Nd=Nd, temp=300)/thickness
        i = np.average(abs(temp-nxc)/nxc)

        nxc = temp

    return nxc
>>>>>>> Stashed changes
