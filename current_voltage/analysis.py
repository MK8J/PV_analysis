
import numpy as np
import scipy.constants as const
import data_operations as rdo
import matplotlib.pylab as plt
import cir_equiv_mdl as ceq


class IV():
    '''
    this is the master class
    everything should go into here
    '''

    mtype_ceq = {
        'LIV': ceq.light_JV,
        'DIV': ceq.dark_JV,
        'SIV': ceq.sudo_JV,
    }

    def __init__(self, I, V, measurement_type, area=1, temp=300):
        self.J = I / area
        self.V = V
        self.measurement_type = measurement_type
        self.temp = temp

    def set_model(self, model='circuit equivalent', **kwargs):
        '''
        set one of the available models
        '''

        if model == 'circuit equivalent':
            self.model = self.mtype_ceq[self.measurement_type](
                temp=self.temp, **kwargs)
            self.fit_model = ceq.fitting_methods(self.model)

    def fit(self, method=None, ax=None):
        '''
        fit a model
        '''
        fit_vals = self.fit_model.fit(self.J, self.V, method=method)

        if ax is not None:
            self.model.get_IV(self.V, comp_vals=fit_vals, ax=ax)
        return fit_vals

    def plot_rawdata(self, ax=None):
        '''
        plot the raw datas
        '''
        if ax is None:
            fig, ax = plt.subplots(1)

        ax.plot(self.V, self.J, '.-')

        if self.measurement_type == 'DIV':
            ax.semilogy()

    def stats(self, data_type='raw data', V=None):
        '''
        returns all dem stats

        inputs:
            data_type: (str: raw data*, model)
                what is analyised to get the stats. 
            V: (optional numpy array)
                if data_type is model, voltages can be provided
                here from which the stats can be determined.
        '''
        if data_type == 'raw data' or V is None:
            V = self.V

        J_dic = {
            'raw data': self.J,
            'model': self.model.get_IV(V)[0],
        }

        stats_dic = {
            'LIV': self._stats_LIV,
            'DIV': self.empty,
            'SIV': self._stats_LIV
        }

        J = J_dic[data_type]

        return stats_dic[self.measurement_type](J, V)

    def _stats_LIV(self, J, V):
        '''
        stats of a LIV curve
        '''
        stats = {}
        stats['Jsc'] = rdo.get_Jsc(J, V)
        stats['Voc'] = rdo.get_Voc(J, V)
        stats['FF'] = rdo.get_FF(J, V)
        stats['MPP'] = rdo.get_MPP(J, V)
        stats['max power'] = rdo.get_maxP(J, V)

        return stats

    def empty(self):
        print 'this is not implimented'
        return None
