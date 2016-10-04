
import sys
import os
import numpy as np

path = os.path.dirname(__file__)
path = '{0}'.format(os.sep).join(path.split(os.sep)[:-3])
sys.path.append(path)

from PV_analysis.current_voltage import cir_equiv_mdl as JV
from PV_analysis.current_voltage import analysis as JV_anal
import matplotlib.pylab as plt


def test_dark_IV_Kaminski1997(ax):
    data = np.genfromtxt(
        'Kaminski1997.csv', names=['V', 'I'], usecols=(0, 1), delimiter=',')
    # makr I into J
    data['I'] /= 10

    data = np.sort(data)

    ax[0].plot(data['V'], data['I'], '.')

    # data = data[data['V'] > 0.1]

    components = [
        'ideal_diode', 'nonideal_diode', 'shunt', 'series']

    dk = JV.dark_JV(components=components)
    fit = JV.fitting_methods(dk)

    # print dk.comps_used.keys()
    for fitting in [fit.least_squares, fit.Rs_via_linalg]:
        fitted_res = fitting(data['V'], data['I'])

        V, I = dk.get_IV(data['V'], ax[0], comp_vals=fitted_res)

        ax[0].plot(V, abs(I - data['I']) / data['I'])

        dk.mv_curve(data['V'], ax[2], comp_vals=fitted_res)


def test_dark_IV(ax):
    data_drk = np.genfromtxt(
        'DarkIV_ym18-w15-4a.drk', skip_header=11, names=['V', 'I'])
    # makr I into J
    data_drk['I'] /= 6.9

    ax[0].plot(data_drk['V'], data_drk['I'], '.')

    data_drk = data_drk[data_drk['V'] > 0.1]

    components = [
        'ideal_diode', 'series', 'nonideal_diode', 'shunt', 'diode_res']

    dk = JV.dark_JV(components=components)
    fit = JV.fitting_methods(dk)

    # print dk.comps_used.keys()
    fitted_res = fit.least_squares(
        data_drk['V'], data_drk['I'], dk.comps_used)

    # V, I = dk.IV_curve_2(data_drk['V'], ax[0], comp_vals=fitted_res)

    # dk.mv_curve(data_drk['V'], ax[2], comp_vals=fitted_res)
    ax[0].semilogy()


def test_light_IV(ax):
    data_lgt = np.genfromtxt(
        'LightIV_ym18-w15-4a.lgt', skip_header=20, names=['V', 'I'])

    # due that current should be negitive
    data_lgt['I'] *= -1
    # make I into J
    data_lgt['I'] /= 6.9

    ax[0].plot(data_lgt['V'], data_lgt['I'] - data_lgt['I'][0], '.-')
    ax[1].plot(data_lgt['V'], data_lgt['I'], '.-')

    components = ['ideal_diode', 'series',
                  'nonideal_diode', 'lgt_gen_cur', 'shunt']

    lgt = JV.light_JV(components=components)

    fit = JV.fitting_methods(lgt)
    print lgt.comps_used['lgt_gen_cur'].params

    for fitting in [fit.Rs_via_linalg, fit.least_squares]:
        try:
            fitted_res = fitting(data_lgt['V'], data_lgt['I'], lgt.comps_used)

            print '\n\n\t', fitted_res, '\n\n'
            V, I = lgt.IV_curve_2(data_lgt['V'], ax[1], comp_vals=fitted_res)
            # lgt.get_IV(data_lgt['V'], ax[1], comp_vals=fitted_res)

            ax[0].plot(V, I - np.amin(I), '--')
            lgt.mv_curve(data_lgt['V'], ax[2], comp_vals=fitted_res)
        except:
            print 'failed'


def test_suns_Voc(ax):
    data = np.genfromtxt(
        'SunIV_ym18-w15-4a.suns', names=['I', 'V'], skip_header=1)

    Jsc = 0.03953679
    # due that current should be negitive
    data['I'] -= 1
    data['I'] *= Jsc
    # make I into J
    # data_lgt['I'] /= 6.9
    data = data[data['I'] < 0]

    ax[0].plot(data['V'], data['I'] + Jsc, '.-')
    ax[1].plot(data['V'], data['I'], '.-')

    components = ['ideal_diode',
                  'nonideal_diode', 'lgt_gen_cur']

    suns = JV.sudo_JV(components=components)
    suns.comps_used['lgt_gen_cur'].params['J'] = Jsc
    # suns.comps_used['lgt_gen_cur'].fitting_params = []

    fit = JV.fitting_methods(suns)
    # print '\n\n\n'
    # print suns.comps_used
    # print [i.params for i in suns.comps_used.values()]
    # print '\n\n\n'

    for fitting in [fit.least_squares]:
        try:
            fitted_res = fitting(data['V'], data['I'], suns.comps_used)

            print '\n\n\t', fitted_res, '\n\n'

            V, I = suns.IV_curve_2(data['V'], ax[1], comp_vals=fitted_res)

            ax[0].plot(V, I - np.amin(I), '-')
            suns.mv_curve(data['V'], ax[2], comp_vals=fitted_res)
        except:
            print 'fitting failed'


def total_test():
    LIV_data = np.genfromtxt(
        'LightIV_ym18-w15-4a.lgt', skip_header=20, names=['V', 'I'])

    LIV_data['I'] *= -1

    LIV = JV_anal.IV(LIV_data['I'], LIV_data['V'], 'LIV', area=6.9)
    LIV.plot_rawdata(ax[0])

    components = ['ideal_diode',
                  'nonideal_diode', 'lgt_gen_cur', 'series', 'shunt']

    LIV.set_model(components=components)

    # print LIV.fit('least squares', ax=ax[0])
    print LIV.fit('least squares_b', ax=ax[0])
    print LIV.stats('model')
    print LIV.stats()
    print LIV.stats('model', np.linspace(0, 0.7, 1000))

    # print LIV.stats()

    # print LIV.stats()


fig, ax = plt.subplots(1, 3, figsize=(16, 6))
# ax[2].set_ylim(0, 5)

# test_dark_IV(ax)
# test_light_IV(ax)
# test_suns_Voc(ax)

# test_dark_IV_Kaminski1997(ax)

total_test()

# ax[0].semilogy()
plt.show()
