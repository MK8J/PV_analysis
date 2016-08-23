
import numpy as np
from semiconductor.recombination import Radiative
from scipy.optimize import curve_fit
import matplotlib.pylab as plt


def calibrate(I_PL, ltc_PC, fitting_index=None):
    '''
    Caculates the doping and wafer dependent constant from
    provided PC and PL raw voltages. No knowledge of the doping is
    required for this method.

    inputs:
        I_PL: (array like, shape m)
            A voltage proportional to the PL intneisty
        ltc_PC: (class)
            A lifetime class from a PC measurment. It is required to have
            thickness, coil constants, temperature, doping_type, all set.
            The PC data should be already background corrected.
        fitting_index: (array like, shape m)
            A mask that selects the data to be fitted.
    '''
    ltc_PC.sample.doping = 1e10

    # is not fitting array, use everything
    if fitting_index is None:
        fitting_index = np.ones(ltc_PC.shape, dtype=bool)

    change = 1
    while change > 0.01:
        ltc_PC._cal_nxc(None)

        Ai, Ndop = calibrate_with_dn_quadratic(
            PL=I_PL[fitting_index],
            nxc=ltc_PC.nxc[fitting_index],
            Na=ltc_PC.sample.Na,
            Nd=ltc_PC.sample.Nd,
            temp=ltc_PC.sample.temp)

        change = abs(ltc_PC.sample.doping - Ndop) / ltc_PC.sample.doping
        ltc_PC.sample.doping = Ndop

    return Ai, Ndop


def calibrate_with_dn_quadratic(PL, nxc, Na, Nd, temp):
    '''
    A function that fits
        PL = A * nxc * (nxc + N_dop) + C

    The with A and N_dop returned. The C value is ignored.
    '''
    _cal_dts = {
        'material': 'Si',
        'temp': 300.,
        'author': None,
        'ni_author': None,
        'Na': 1,
        'Nd': 1e16,
    }

    B = Radiative(material='Si', temp=temp, Na=Na, Nd=Nd).get_B(nxc)
    # vals = np.polyfit(nxc, PL / B, 2, w=1. / PL)

    def func(nxc, Ai, Ndop):
        return Ai * nxc * (nxc + Ndop)

    popt, pcov = curve_fit(func, nxc, PL / B, p0=(1e-19, 1e10), sigma=PL / B)

    Ai, Ndop = popt[0], popt[1]

    # Ai = vals[0]
    # Ndop = vals[1] / Ai

    return Ai, Ndop
