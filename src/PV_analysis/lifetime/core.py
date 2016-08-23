

import numpy as np
import numbers
import matplotlib.pylab as plt
import sys
import os

import scipy.constants as C
# sys.path.append(r'D:\Dropbox\CommonCode\semiconductor\src')
# sys.path.append(r'/home/shanoot/Dropbox/CommonCode/semiconductor/src')
from semiconductor.recombination import Intrinsic as IntTau
from semiconductor.material import IntrinsicCarrierDensity
from semiconductor.material import BandGapNarrowing
from semiconductor.electrical import Mobility
from PV_analysis.common import IO, sample, background_correction


def tau_ext_from_tau_eff(self, tau_eff, min_car_den, Na, Nd, **kwargs):
    '''
    Determines the extrinsic carrier lifetime,
    by subtraction of the intrinsic carrier lifetime

    inputs:
        tau_eff,
            the effective carrier lifetime  in s
        min_car_den:
            the minorty carrier density in cm^-3
        Na:
            number of acceptor dopants (p-type)
        Nd:
            number of donor dopants (n-type)
        **kwargs
            any special parameters for the intrinsic lifetime caculation
    '''

    tau = IntTau(material='Si', **kwargs)
    return 1. / (1. / tau_eff - 1. / tau.tau(min_car_den, Na, Nd))


def murphy_plot(nxc, tau_ext, ax, label=None):
    '''
    plots lifetime data the Murphy way (10.1063/1.4725475)
    This is particularly useful for when looking for
    SRH defects
    '''

    ax.plot(nxc, tau_ext, ':',
            label=None)

    ax.legend(loc=0)
    ax.set_xlabel('n/p')
    ax.set_ylabel('$\tau_{eff}$ (s)')


def caltau_generalised(nxc, gen, time, other=0):
    '''
    caculates the lifetime with the generatlised method
    inputs:
        nxc: (array like (cm^-3))
            number of excess carrier density
        gen: (array like (cm^-3))
            number of photons. Required to the be same shape as nxc
        time: (array like (s))
            the time for which the excess carrier density and generation is
            reported
    output:
        tau: (array like (s))
            the effective lifetime in seconds
    '''

    dndt = np.gradient(nxc, time[2] - time[1])

    return nxc / (gen - dndt - other)


def caltau_steadystate(nxc, gen, *args):
    '''
    caculates the lifetime with the generatlised method
    inputs:
        nxc: (array like (cm^-3))
            number of excess carrier density
        gen: (array like (cm^-3))
            number of photons. Required to the be same shape as nxc
    output:
        tau: (array like (s))
            the effective lifetime in secondsds
    '''

    return nxc / (gen)


def caltau_transient(nxc, time, *args):
    '''
    caculates the lifetime using the assuming a transient
    inputs:
        nxc: (array like (cm^-3))
            number of excess carrier density
        time: (array like (s))
            the time for which the excess carrier density and generation is
            reported
    output:
        tau: (array like (s))
            the effective lifetime in seconds
    '''

    dndt = np.gradient(nxc, time)

    return nxc / (dndt)


class lifetime():

    # measured values
    time = None
    gen = None
    nxc = None

    # caculated value
    tau = None

    # ensure you know what the file is
    file_name = None
    sample_name = None

    # material properties/analysis options
    analysis_options = {
        'analysis': 'generalised',
        'bgc_side': 'front',
        'bgc_type': 'points',
        'bgc_value': 100,
        'ni': None,
        'ni_eff': None,
        'auger': None,
        'D_ambi': None
    }

    _analsis_methods_dic = {
        'generalised': caltau_generalised,
        'transient': caltau_steadystate,
        'steadystate': caltau_transient,
    }

    _warnings = True
    _type = ''  # a saving extension

    def __init__(self, **kwargs):
        self.attr = kwargs
        self.sample = sample()
        pass

    def _bg_correct(self, data):
        return background_correction.timeseries(
            data,
            self.analysis_options['bgc_value'],
            self.analysis_options['bgc_type'],
            self.analysis_options['bgc_side'])

    def _cal_lifetime(self, analysis=None, other=0):
        self.analysis_options[
            'analysis'] = analysis or self.analysis_options['analysis']

        self.tau = self._analsis_methods_dic[self.analysis_options['analysis']](
            nxc=self.nxc, gen=self.gen / (
                self.sample.absorptance * self.sample.thickness
            ), time=self.time, other=other)

    @property
    def iVoc(self):
        '''
        returns the open circuit voltage of the device
        '''
        return const.k * self.sample.temp / const.e * np.log(
            self.nxc * (self.nxc + self.sample.doping) / self.ni**2)

    @property
    def ni(self):
        if isinstance(self.analysis_options['ni'], numbers.Number):
            _ni = self.analysis_options['ni']
        elif isinstance(self.analysis_options['ni'], np.ndarray):
            _ni = self.analysis_options['ni']
        else:
            _ni = IntrinsicCarrierDensity(
                material='Si', temp=self.sample.temp,
                author=self.analysis_options['ni']
            ).update()
        return _ni

    @ni.setter
    def ni(self, val):
        self.analysis_options['ni'] = val

    @property
    def ni_eff(self):
        if isinstance(self.analysis_options['ni_eff'], numbers.Number):
            _ni = self.analysis_options['ni_eff']
        elif isinstance(self.analysis_options['ni_eff'], np.ndarray):
            _ni = self.analysis_options['ni_eff']
        else:
            _ni = self.ni * BandGapNarrowing(
                material='Si',
                temp=self.sample.temp,
                author=self.analysis_options['ni_eff'],
                nxc=self.nxc,
                Na=self.sample.Na,
                Nd=self.sample.Nd
            ).ni_multiplier()
            print(self.ni, _ni)
        return _ni

    @ni_eff.setter
    def ni_eff(self, val):
        self.analysis_options['ni_eff'] = val

    @property
    def mobility_model(self):
        return self.analysis_options['mobility'],

    @mobility_model.setter
    def mobility_model(self, value):
        self.analysis_options['mobility'] = value

    @property
    def D_ambi(self):
        if isinstance(self.analysis_options['D_ambi'], numbers.Number):
            _D = self.analysis_options['D_ambi']
        elif isinstance(self.analysis_options['D_ambi'], np.ndarray):
            _D = self.analysis_options['D_ambi']
        else:
            vt = C.k / C.e * self.sample.temp
            if self.sample.dopant_type == 'n-type':
                _D = vt * Mobility(
                    author=self.analysis_options['mobility'],
                    material='Si', temp=self.sample.temp,
                    nxc=self.nxc, Na=self.sample.Na, Nd=self.sample.Nd
                ).electron_mobility()
            elif self.sample.dopant_type == 'p-type':
                _D = vt * Mobility(
                    author=self.analysis_options['mobility'],
                    material='Si', temp=self.sample.temp,
                    nxc=self.nxc, Na=self.sample.Na, Nd=self.sample.Nd
                ).hole_mobility()
            else:
                print('sample dopant_type incorrected set as',
                      self.sample.dopant_type)
                print('Needs to be' + 'n - type or p - type')
        return _D

    @D_ambi.setter
    def D_ambi(self, value):
        self.analysis_options['D_ambi'] = val

    # TODO:
    # this needs to be changed from auger to intrinsic_lifetime

    @property
    def auger(self):
        if isinstance(self.analysis_options['auger'], np.ndarray):
            _auger = self.analysis_options['auger']
        else:
            _auger = IntTau(
                material='Si',
                temp=self.sample.temp,
                Na=self.sample.Na,
                Nd=self.sample.Nd,
                author=self.analysis_options['auger'],
                rad_author=None,
                aug_author=self.analysis_options['auger'],
                ni_author=self.analysis_options['ni']
            ).tau(nxc=self.nxc)

        return _auger

    @auger.setter
    def auger(self, val):
        self.analysis_options['auger'] = val

    @property
    def attrs(self):
        return {
            'doping': self.sample.doping,
            'thickness': self.sample.thickness,
            'dopant type': self.sample.dopant_type,
            'tau': self.tau,
            'nxc': self.nxc,
        }

    @attrs.setter
    def attrs(self, kwargs):
        self.other_inf = {}
        assert type(kwargs) == dict
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.sample, key):
                setattr(self.sample, key, value)
            else:
                if self._warnings:
                    print('Attribute {0} not found'.format(key))
                    print('Attribute set in dic other_inf')
                self.other_inf[key] = value

    def crop_nxc(self, min_nxc, max_nxc):
        '''
        Crops the data to a provided carrier density
        '''

        index = self.nxc > min_nxc
        index *= self.nxc < max_nxc

        self.nxc = self.nxc[index]
        self.tau = self.tau[index]
        self.gen = self.gen[index]
        if isinstance(self.analysis_options['auger'], np.ndarray):
            self.auger = self.auger[index]
        if isinstance(self.analysis_options['ni_eff'], np.ndarray):
            self.ni = self.ni[index]

    def save(self):

        x = np.core.records.fromarrays(
            [self.time, self.nxc, self.tau, self.gen],
            names='time,nxc,tau, gen')

        if self.file_name.count('.') == 1:
            save_name = self.file_name.split('.')[0] + '_' + self.type
        else:
            save_name = self.file_name
        IO.save_named_data(self.file_name, )
