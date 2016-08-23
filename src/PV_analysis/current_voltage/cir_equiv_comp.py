
import numpy as np
import scipy.constants as const
import scipy.optimize as opt


class component():

    def __init__(self, temp=300, params=None):
        self.temp = temp
        if params is not None:
            self.params = params

    def Vt(self, temp):

        if temp is None:
            temp = self.temp

        return const.k * temp / const.e


class diode(component):
    params = {
        'J0': 1e-13,
        'n': 1}

    fitting_params = [
        'J0'
    ]

    units = {
        'J0': 'A/cm^-2',
        'n': 'None'
    }

    def _get_lin_params(self, value):

        par = self.params.copy()
        par['J0'] = value
        return par

    def _set_lin_params(self, value):
        self.params['J0'] = value


    def _lin_sys_comp(self, V, temp=None):
        '''
        provides the diodes linear component. 
        For this it is assumed that the ideality factor is constant
        '''

        return np.exp(V / self.params['n'] / self.Vt(temp)) - 1.

    def current(self, V, params=None, temp=None):
        '''
        returns the current through a diode
        '''
        if params is None:
            params = self.params

        return params['J0'] * \
            (np.exp(V / self.Vt(temp) / params['n']) - 1)

    def djdv(self, V, params=None, temp=None):
        '''
        returns the derivative of the component
        '''

        if params is None:
            params = self.params

        return params['J0'] * (
            np.exp(V / self.Vt(temp) / params['n'])
        ) / params['n'] / self.Vt(temp)

    def voltage(self, I, params=None, temp=None):
        '''
        returns the voltage drop over a diode
        '''
        if params is None:
            params = self.params

        return params['n'] * self.Vt(temp) * \
            np.log(J / params['J0'] + 1)


class diode_resistor(component):
    params = {
        'J0': 1e-13,
        'n': 2,
        'R': 10,
    }

    fitting_params = [
        'J0', 'R'
    ]

    units = {
        'J0': 'A/cm^-2',
        'n': 'None',
        'R': 'Ohms/cm^-2'
    }

    def current(self, V, params=None, temp=None):
        '''
        returns the current through a diode
        '''
        if params is None:
            params = self.params

        J = np.ones(V.shape)

        index = V < 0
        J[index] = -params['J0']
        index2 = V > 0

        for i in range(V[index2].shape[0]):
            J[i] = opt.brentq(
                self._itterative_IV, 0, 1, args=(V[index2][i], temp))

        return J

    def _itterative_IV(self, J, V, temp):
        '''
        this current voltage curve can not be solved
        directly, this provides the relationship between the two
        '''
        return (-self.params['n'] * self.Vt(temp) *
                np.log(J / self.params['J0'] + 1.) -
                self.params['R'] * J) + V

    def djdv(self, V, params=None, temp=None):
        '''
        returns the derivative of the component
        '''

        if params is None:
            params = self.params

        if temp is None:
            temp = self.temp

        var = self.current(V) + params['J0']

        return var / (
            params['R'] * var + params['n'] * self.Vt(temp))


class resistor(component):
    fitting_params = [
        'R'
    ]


    def current(self, V, params=None, temp=None):
        '''
        returns the current through a resistor
        '''
        if params is None:
            params = self.params

        return V / params['R']

    def djdv(self, V, params=None, temp=None):
        '''
        returns the derivative of the component
        '''
        if params is None:
            params = self.params

        return np.ones(V.shape[0]) / params['R']

    def voltage(self, J, params=None, temp=None):
        '''
        returns the voltage drop over a resistor
        '''
        if params is None:
            params = self.params

        return J * params['R']


class resistor_parallel(resistor):
    params = {
        'R': 1000,
    }

    units = {
        'R': 'Ohms/cm^-2'
    }

    def _set_lin_params(self, value):
        self.params['R'] = 1./value

    def _get_lin_params(self, value):

        par = self.params.copy()
        par['R'] = 1. / value
        return par

    def _lin_sys_comp(self, V, params=None):
        return V


class resistor_series(resistor):
    params = {
        'R': 1,
    }

    units = {
        'R': 'Ohms'
    }

    def _get_lin_params(self, value):

        par = self.params.copy()
        par['R'] = value
        return par


class current_source(component):
    params = {
        'J': 0.038}
    units = {'J': 'A/cm^-2'}

    fitting_params = [
        'J'
    ]

    def _get_lin_params(self, value):

        par = self.params.copy()
        par['J'] = value
        return par

    def _set_lin_params(self, value):
        self.params['J'] = value


    def _lin_sys_comp(self, V, temp=None):

        return -1*np.ones(V.shape)

    def current(self, V=None, params=None, temp=None):
        '''
        returns the current through a current source
        '''

        if params is None:
            params = self.params

        return -1*params['J'] * np.ones(V.shape[0])

    def voltage(self, I, params=None, temp=None):
        '''
        returns the voltage drop over a current source
        '''
        print 'Does this make sense?'
        print 'What are you trying to do?'
        print 'This isn\'t magic'

        pass

    def djdv(self, V, params=None, temp=None):
        '''
        returns the derivative of the component
        '''

        return np.zeros(V.shape[0])

# some other things that could be added

# def transistor_V(V, J01):
#     '''
#     returns the current through a transistor
#     '''
#     J = np.ones(V.shape)
#     for i in range(V.shape[0]):
#         J[i] = opt.brentq(
#             transistor_zerofinding, 1e-9, 1e-3, args=(V[i], J01, J01))
#     return J


# def transistor_zerofinding(J, V, J01, J02):
#     V1 = doide_J(J, J01, 1)
#     V2 = doide_J(-J, J02, 1)

#     return V1 - V2 - V


# def reverse_doide_V(V, J0, n=2, temp=300):
#     '''
#     returns the current in a reverse diode
#     '''
#     return abs(doide_V(-V, J0, n, temp))


# def reverse_doide_I(V, J0, n=2, temp=300):
#     '''
#     returns the current in a reverse diode
#     '''
#     return abs(doide_J(-V, J0, n, temp))



