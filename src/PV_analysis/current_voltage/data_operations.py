
import numpy as np
import scipy.constants as const
import matplotlib.pylab as plt


def Vt(temp=300):

    return const.k * temp / const.e


def get_mV(I, V, area=1, Jsc=0, temp=300):
    '''
    return the local ideality factor, voltage curve
    for a constant voltage spacing
    using the second order difference method
    '''

    J = I / area + Jsc

    dlnJdV = np.gradient(np.log(J), varargs=V[1] - V[0])

    # doing the derivative of the log, provides a slightly
    # better noise reduction
    m = 1. / dlnJdV / Vt(temp)

    return m


def get_PV(I, V, area=1):
    '''
    return the power voltage curve
    '''

    J = -I / area
    P = J * V

    return P


def get_maxP(I, V, area=1):
    '''
    return the max power
    '''

    P = get_PV(I, V, area)

    return np.amax(P)


def get_MPP(I, V, area=1):
    '''
    return the current and voltage at maximum power point
    '''
    J = I / area

    P = get_PV(I, V, area)

    index = P == np.max(P)

    return abs(J[index][0]), V[index][0]


def get_FF(I, V, Jsc=None, Voc=None, Imp=None, Vmp=None):
    '''
    caculate the fill factor
    '''
    if Imp is None or Vmp is None:
        Imp, Vmp = get_MPP(I, V)
    if Jsc is None:
        Jsc = get_Jsc(I, V)
    if Voc is None:
        Voc = get_Voc(I, V)

    return Imp * Vmp / (Jsc * Voc)


def get_Jsc(I, V, area=1):
    '''
    caculate the Jsc
    '''
    Isc = -np.interp(0, V, I)

    return Isc / area


def get_Voc(I, V):
    '''
    caculate the Voc
    '''

    Voc = np.interp(0, I, V)

    return Voc
