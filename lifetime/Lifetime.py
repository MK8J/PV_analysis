

import numpy as np
import matplotlib.pylab as plt
import sys
import os

sys.path.append(r'C:\Users\z3186867\Dropbox\CommonCode')
# sys.path.append(r'C:\Users\mattias\Dropbox\CommonCode')
sys.path.append(r'/home/shanoot/Dropbox/CommonCode')

# import semiconductor.optical.opticalproperties as OP
import SintonInstruments_WCT as WCT

import semiconductor.material.ni as niclass
from semiconductor.recombination.Intrinsic import Intrinsic as IntTau

from scipy import interpolate
from scipy import optimize
import scipy.constants as const


class CommonLifetimeFunctions():

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

    def _plot_murfy(self, ax, label=None):
        '''
        plots lifetime data the Murphy way (10.1063/1.4725475)
        This is particularly useful for when looking for
        SRH defects
        '''

        ax.plot(self.nxc, self.tau_ext, ':',
                label=None)

        ax.legend(loc=0)
        ax.set_xlabel('n/p')
        ax.set_ylabel('$\tau_{eff}$ (s)')


class fit_sintons(CommonLifetimeFunctions):


    def __init__(self, fnames, num_SRH, Jo):
        '''
                fnames: (str or list)
                        The file path to a Sinton file or a list
                        of file paths all to be fitted at the same time
                num_SRH: (int)
                        The number of SRH defects to be fitted
                Jo: (bool)
                        If Jo should be fitted
                sinton_version: (str ['b42014','af2014'])
                        A string that tells the if the Sinton file is older that 2014
        '''

        self.fnames = fnames
        self.num_SRH = num_SRH
        self.Jo = Jo

        self._check_inputs()
        self._input()

    def _check_inputs(self):
        # if fnames is string make list
        if isinstance(self.fnames, str):
            self.fnames = [self.fnames]

        # make sure files exist
        for fname in self.fnames[:]:
            if not os.path.isfile(fname):
                print 'File path does not exist: ', fname,
                cont = raw_input('File removed from list and continue?\n')
                if cont == 'n':
                    sys.exit("Program stopped by user")
                elif cont == 'y':
                    self.fnames.remove(fname)

        assert isinstance(self.num_SRH, int)
        assert isinstance(self.Jo, bool)

    def _input(self):

        self.m_data = []
        self.input_data = []
        for fname in self.fnames[:]:
            self.m_data.append(WCT.extract_raw_data(fname))
            self.input_data.append(WCT.extract_usr_data(fname))

    def _plot(self):
        fig, ax  = plt.subplots()
        for data, inf in zip(self.m_data, self.input_data):
            print inf['thickness'], inf['optical_constant'], inf['doping'], inf['resisitivity'], inf['m_resisitivity']

            ax.plot(data['Minority Carrier Density'], data['Tau (sec)'], label=inf['wafer_name'])
            # ax.plot(data['Apparent CD'][2:-2], data['Tau (sec)'][2:-2], label=inf['wafer_name'])


        ax.legend(loc=0)
        ax.loglog()
        ax.set_xlabel('Excess carriers')
        ax.set_ylabel('Lifetime (s)')

        plt.show()

    def plot_murfy(self):
        fig, ax  = plt.subplots(1)
        for data, inf in zip(self.m_data, self.input_data):
            self.nxc = data['Minority Carrier Density']
            self.tau_ext = self.tau_ext_from_tau_eff(data['Tau (sec)'], data['Minority Carrier Density'], 0, inf['doping'])
            self._plot_murfy(ax)



class CurrentRecombinationPrefactorModels():

    def __init__(self, matterial='Si', model_author=None):
        self.ni = niclass.IntrinsicCarrierDensity(matterial)
        self.ni.update_ni()

    def J0_function(self, maj_car_den, Jo, tau_bulk, thickness=None, ni='constant'):
        if thickness is None:
            thickness = self.thickness

        if ni == 'constant':
            itau = Jo * maj_car_den / const.e / \
                self.ni.ni ** 2 / thickness + 1. / tau_bulk
            # const.e * self.ni.ni ** 2 * thickness / (maj_car_den) + tau_bulk

        return 1. / itau

    def J0_symetric(self, tau_ext, min_car_den, doping, thickness, Plot):
        '''
        Calculates J0 fro a double side diffused surface
        taken from Cuevas2004

        Assumes a constant ni with delta n.
        '''

        self.thickness = thickness

        popt, pcov = optimize.curve_fit(
            self.J0_function, min_car_den + doping, tau_ext, p0=(1e-15, 1e-6))

        if Plot:
            plt.plot(
                min_car_den, 1. / tau_ext, 'bo', label='Raw data')

            itau_fitted = 1. / self.J0_function(min_car_den + doping, *popt)

            plt.plot(
                min_car_den, itau_fitted, 'r-', label='Fitted data')

        # Assig J0 to both sides

        return popt

    def tau_EmitterRecombination(self):
        return 1. / self.Cuevas2004()

    def itau_EmitterRecombination(self):
        return 1. / self.tau_EmitterRecombination()

    def PlotAll(self, ax):

        ax.plot(self.Deltan, self.tau_EmitterRecombination(), ':',
                label='Emitter Recombination:' + str(self.J0e))

        legend(loc=0)
        loglog()
        ax.set_xlabel('$\Delta$ n (cm$^{-3}$)')
        ax.set_ylabel('$\Tau_eff$ (s)')
        # show()


class CaculateSurfaceRecombiatnion(CommonLifetimeFunctions):

    def __init__(self):
        self.JoModels = EmitterSaturationCurrentDensityModels()

    def Calculate_J0(self, tau_eff, min_car_den, Na, Nd, thickness, model='', Plot=True):
        '''
        Caculates Jo in A/cm2
            inputs:
                tau:
                    the effective minority carrier lifetime in seconcds
                min_car_den:
                    the minorty carrier density in cm^-3
                Na:
                    number of acceptor dopants (p-type)
                Nd:
                    number of donor dopants (n-type)
                thickness:
                    thickness of the sample in cm
                model:
                    can choose from "symetric"
                Plot:
                    will plot the data and the fit
        '''

        tau_ext = self.tau_ext_from_tau_eff(tau_eff, min_car_den, Na, Nd)

        doping = np.amax([Na, Nd]) - np.amin([Na, Nd])

        # try:
        J0, bulk_tau = getattr(
            self.JoModels, 'J0_' + model)(tau_ext, min_car_den, doping, thickness, Plot)

        # except:
        #     print 'That is not a model'

        return J0, bulk_tau


class Test_Jo(CommonLifetimeFunctions):

    def test(self):
        data = np.genfromtxt(
            r'Example_lifeitme.dat', names=True, delimiter='\t')
        print data.dtype.names
        #('Time_s', 'Photovoltage', 'Reference_Voltage', 'Apparent_CD', 'Tau_sec', '1Tau_corrected')

        Na, Nd, width = 8.1e15, 0, 0.018

        data = data[data['Tau_sec'] > .0]
        data = data[data['Minority_Carrier_Density'] > .6e16]

        # tau_ext = self.tau_ext_from_tau_eff(
        #     data['Tau_sec'], data['Minority_Carrier_Density'], Na, Nd)

        ''' Fit data'''
        Jo = CaculateSurfaceRecombiatnion()

        fited_jo, bulk_tau = Jo.Calculate_J0(
            data['Tau_sec'], data['Minority_Carrier_Density'], Na, Nd, width, model='symetric')

        print fited_jo, bulk_tau * 1e6

        Sinton_Fit = CurrentRecombinationPrefactorModels().J0_function(
            data['Minority_Carrier_Density'], 1.14e-13, 33.7e-6, 0.018)
        taus = CurrentRecombinationPrefactorModels().J0_function(
            data['Minority_Carrier_Density'], fited_jo, bulk_tau, 0.018)

        plt.plot(
            data['Minority_Carrier_Density'], 1. / Sinton_Fit, 'g--')

        plt.plot(
            data['Minority_Carrier_Density'], data['1Tau_corrected'], 'r.')

        plt.plot(
            data['Minority_Carrier_Density'], 1. / data['Tau_sec'], 'g.')

        plt.ylim(0, 120000)

        plt.show()

if __name__ == "__main__":
    # test = Test_Jo()
    # test.test()
    fname = r'/home/shanoot/Downloads/49-75-iVoc.xlsm'


    fnames = [
    r'D:\ownCloud1\UNSW\PhD\backup\Measurements\PHD\QSS PC\WCT-120\Karola\49-75-iVoc.xlsm',
    # r'/home/shanoot/Temp, while own clod updates/B2-SiNx-only.xlsm',
    # r'/home/shanoot/Temp, while own clod updates/B99-Diff-SiNx.xlsm',
    # r'/home/shanoot/Temp, while own clod updates/B100-SiNx-Only.xlsm',
    # r'/home/shanoot/Temp, while own clod updates/G1-Diff-SiNx.xlsm',
    # r'/home/shanoot/Temp, while own clod updates/G2-SiNx-Only.xlsm',
    # r'/home/shanoot/Temp, while own clod updates/G99-Diff-SiNx.xlsm',
    # r'/home/shanoot/Temp, while own clod updates/G100-SiNx-only.xlsm',
    ]
    a = fit_sintons(fnames, 0, False, )
    a._plot()

    a.plot_murfy()
    plt.show()
