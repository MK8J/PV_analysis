
import sys
import os
import numpy as np
import matplotlib.pylab as plt

path = os.path.dirname(__file__)
path = '{0}'.format(os.sep).join(path.split(os.sep)[:-3])
sys.path.append(path)

from PV_analysis.current_voltage import cir_equiv_mdl as JV_mod
from PV_analysis.current_voltage.analysis import IV


fname = r'YM18-9-3B SunsVoc.dat'
data = np.genfromtxt(fname, delimiter='\t', names=True)

data['J'] *= -1
data = data[data['J'] < 0.01]

fig, ax = plt.subplots(1, 2, figsize=(16, 6))


JV = IV(data['J'], data['V'], 'SIV', 1, 23.6 + 273.15)
JV.plot_rawdata(ax[0])

two_diode = ['ideal_diode',
             'nonideal_diode', 'lgt_gen_cur', 'shunt']

two_diode_dark = ['ideal_diode',
                  'nonideal_diode', 'shunt']

one_diode = ['ideal_diode',
             'lgt_gen_cur', 'shunt']

three_diode_dark = ['ideal_diode', 'diode_res',
                    'nonideal_diode', 'shunt']

JV.set_model(components=two_diode)
fit = JV.fit(ax=ax[0], method='least squares')
print JV.stats('model')

JV.J += fit['lgt_gen_cur']['J']

JV.plot_rawdata(ax[1])

JV.set_model(components=two_diode_dark)

print fit
print JV.fit(ax=ax[1], method='least squares_b')

# plt.legend(['Raw data', 'diode', '2 diode'], loc=0)
ax[1].semilogy()
for axes in ax:
    axes.set_xlabel('Voltage (V)')
    axes.set_ylabel('Current (A)')


plt.show()
