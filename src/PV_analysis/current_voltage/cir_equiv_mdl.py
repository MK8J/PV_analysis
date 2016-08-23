
import numpy as np
import numpy.linalg as nalg
import matplotlib.pylab as plt
import time as t
import scipy.optimize as opt

import cir_equiv_comp as comp
import data_operations as do


# To Do
# there is an issue when fitting light Iv and Suns Voc
# currently I have a weighted fit to the small current values
# which is needed for dark IV fits, though this
# is not the case for light IV and Suns Voc. Hence, the problem

def _terminal_to_junction_voltage(Vt, J, comps_used, Vj_g_Vt):
    '''
    coverts the terminal voltage to the junction voltage
    just helps you make sure your not making a mistake

    inputs:
        Vt: (array)
            terminal voltage
        J: (array)
            terminal current
        comps_used: (dic)
            a dictionary of the components used

    outputs:
        Vj: (array)
            the junction voltage
    '''

    # check if there is a series component else ignore
    if 'series' in comps_used:
        Vj = Vt - (-1)**(Vj_g_Vt) * \
            comps_used['series'].voltage(np.abs(J))
    else:
        Vj = np.copy(Vt)
    return Vj


def _junction_to_terminal_voltage(Vj, J, comps_used, Vj_g_Vt):
    '''
    coverts the junction voltage to the terminal voltage
    just helps you make sure your not making a mistake

    inputs:
        Vj: (array)
            terminal voltage
        J: (array)
            terminal current
        comps_used: (dic)
            a dictionary of the components used

    outputs:
        Vt: (array)
            the junction voltage
    '''
    # check if there is a series component else ignore
    if 'series' in comps_used:
        Vt = Vj + (-1)**(Vj_g_Vt) * \
            comps_used['series'].voltage(np.abs(J))
    else:
        Vt = np.copy(Vj)
    return Vt


class _general_JV():
    '''
    A base class for simulating IV curves
    '''

    # a dictionary of the list of all
    # components
    comps_avail = {
        'diode': comp.diode(),
        'ideal_diode': comp.diode(),
        'nonideal_diode': comp.diode(),
        'shunt': comp.resistor_parallel(),
        'series': comp.resistor_series(),
        'lgt_gen_cur': comp.current_source(),
        'diode_res': comp.diode_resistor(),
        # 'reverse_doide': comp.reverse_doide,
        # 'transistor': comp.transistor_V,
    }

    # default values for simulations
    comp_vals = {
        'diode': None,
        'ideal_diode': {'J0': 1e-13, 'n': 1},
        'nonideal_diode': {'J0': 1e-8, 'n': 2},
        'shunt': {'R': 1e3},
        'series': {'R': 1},
        'lgt_gen_cur': {'J': 0.040},
        'diode_res': {'J0': 1e-5, 'n': 2, 'R': 10}
        # 'reverse_doide': [1e-13],
        # 'transistor': [1e-8],
    }

    component = ['ideal_diode', 'nonideal_diode',
                 'shunt', 'series', 'lgt_gen_cur']

    # this represents if the terminal voltage
    # is greater than the junction voltage
    Vj_g_Vt = False

    def __init__(self, components=None, temp=300., area=1):
        if components is None:
            components = self.component

        # this is a list of the name, followed by the correct handle
        self.comps_used = self._init_comps(components)
        self.temp = 300

    def _init_comps(self, comps_used):
        '''
        initializes components
        '''

        # removes bad component names
        comps_used = self._check_comps(comps_used)

        # makes a dic of names and handles
        comps_used = {i: self.comps_avail[i] for i in comps_used}

        # makes the default values
        comps_used = self._vals_to_comps(comps_used)

        return comps_used

    def _vals_to_comps(self, comps_used, comp_vals=None):
        '''
        sets the default parameter values
        '''
        if comp_vals is None:
            comp_vals = self.comp_vals
        # else:
        #     comp_vals['series']['R'] *= 1

        for component, handel in comps_used.items():

            handel.params = comp_vals[component]

        return comps_used

    def _check_comps(self, comps_used):
        '''
        checks if the components have a model
        '''
        for i in set(comps_used):
            if i not in self.comps_avail.keys():
                index = comps_used.index(i)
                del comps_used[index]
                print str(i) + 'is not a valid component and has been deleted'

        return comps_used

    def _plot_constitues_IV(self, V, ax):
        '''
        plots the current at a given voltage
        for the current components

        This is currently a bit of a hack, and will 
        cause poor voltage scales for devices with large Rs
        '''
        J = np.zeros(V.shape)

        for comp, handle in self.comps_used.items():
            if comp is not 'series':
                J += handle.current(V)

        # add on the series voltage drop
        if 'series' in self.comps_used.keys():

            Vt = V + (-1)**(self.Vj_g_Vt) * \
                self.comps_used['series'].voltage(J)
        else:
            Vt = V

        for comp, handle in self.comps_used.items():
            if comp is not 'series':
                J = handle.current(V)
                ax.plot(Vt, J, '--')

        return np.vstack((Vt, J))

    def print_compants_values(self):
        '''
        prints the active components and their values
        '''

        # make header
        print 'Fitted Componants'
        print '--------------------'
        print 'Componant: parameter \t value \t unit\n'

        # print values
        for component, handle in self.comps_used.items():
            print '\t' + component + ':'
            for param, value in handle.params.items():
                print '\t\t {0}: {1:.2e} '.format(param, value),
                print handle.units[param]

    def get_mv(self, V, ax=None, adjust=True, comp_vals=None):
        '''
        returns a series resistance free m-V curve from the
        model
        '''
        if comp_vals is not None:
            self._vals_to_comps(self.comps_used, comp_vals)

        dJdVj = np.zeros(V.shape)
        dJdV = np.zeros(V.shape)

        V, J = self.IV_curve_2(V, ax=None, adjust=True, comp_vals=None)

        Vj = _terminal_to_junction_voltage(J, V, self.comps_used, self.Vj_g_Vt)

        for component, handle in self.comps_used.items():
            if component is not 'series':
                dJdVj += handle.djdv(Vj)

        dJdV = dJdVj * np.gradient(Vj) / np.gradient(V)

        if 'lgt_gen_cur' in self.comps_used.keys():
            Jsc = self.comps_used['lgt_gen_cur'].params['J']
        else:
            Jsc = 0

        print Jsc

        m = (J + Jsc) / dJdV / handle.Vt(None)

        if ax is not None:
            ax.plot(V, m, '-')

        return np.vstack((V, m))

    def _IV_at_Vj(self, Vj):
        '''
        a private function that calculates the current
        out at a given voltage for the current components
        '''
        J = np.zeros(Vj.shape)

        for component, handle in self.comps_used.items():
            if component is not 'series':
                J += handle.current(Vj)

        return J

    def get_IV(self, V, ax=None, comp_vals=None, method='iterative'):
        '''
        calculates and returns an IV curve given an input voltage.
            inputs:
                V: (array)
                    the terminal voltage
                ax: (optional matplotlib.pylab axis)
                    if a axis is passed, a plot will be produced
                comp_val: (optional, dic)
                    a dictionary of component names, and parameters.
                    These will then be used over the current values
                method: (optional: iterative*, Rsshift)
                    The method by which the IV curve is determined.
                    The method differ owing to Rs. The iterative method
                    iterates the Vj to find the same terminal voltage as
                    provided. The Rs shift, does not return the same IV
                    at the input voltage but rather it is shifted by the
                    voltage Rs * J.
            output:
                I, V

        '''

        IV_dic = {
            'Rsshift': self._IV_curve_Rsshift,
            'iterative': self._IV_curve_itterative,
        }

        return IV_dic[method](V, ax, comp_vals)

    def _IV_curve_Rsshift(self, V, ax=None, comp_vals=None):
        '''
        calculates the the IV curve

        note the output voltage differs from the input
        '''

        # if values are provided, send them to the components
        if comp_vals is not None:
            self._vals_to_comps(self.comps_used, comp_vals)

        # we start with 0 current and built it up
        J = np.zeros(V.shape)

        # get the current through the parallel components
        for component, handle in self.comps_used.items():
            if component is not 'series':
                J += handle.current(V)

        # add on the series voltage drop
        if 'series' in self.comps_used.keys():
            Vt = _junction_to_terminal_voltage(
                V, J, self.comps_used, self.Vj_g_Vt)

        else:
            Vt = V

        if ax is not None:
            ax.plot(Vt, J, '-')

        return np.vstack((J, Vt))

    def _IV_curve_itterative(self, V, ax=None, comp_vals=None):
        '''
        calculates the the IV curve
        through iterating the voltage drop across the resistor

        note the output voltage differs from the input
        '''

        # if values are provided, send them to the components
        if comp_vals is not None:
            self._vals_to_comps(self.comps_used, comp_vals)

        # get rs free J
        J = self._IV_at_Vj(V)

        i = 0
        # if series res iterate to find real rs
        if 'series' in self.comps_used.keys():
            J_temp = np.copy(J)
            changeV = self.comps_used['series'].voltage(np.amax(J_temp))
            # print changeV

            while changeV > 0.0001:
                i += 1
                # print i, '\r', changeV,
                # we start with 0 current and built it up
                J = np.zeros(V.shape)

                # calculate voltage drop
                Vj = _terminal_to_junction_voltage(
                    V, J_temp, self.comps_used, self.Vj_g_Vt)

                # get the current through the parallel components
                J = self._IV_at_Vj(Vj)

                changeV = abs(self.comps_used['series'].voltage(
                    np.amax(J_temp) - np.amax(J)))
                J_temp = (J - J_temp) * 0.5 + J_temp

        # print i
        if ax is not None:
            ax.plot(V, J, '-')
            # print np.vstack((V, J))
            # ax.semilogy()
        return np.vstack((J, V))

    def get_Jsc(self, J, V):
        '''
        calculates the Jsc
        '''

        print 'di i make it?'
        Jsc, V = self.IV_curve_itterative(np.zeros(1))

        return abs(Jsc)


class light_JV(_general_JV):
    comps_avail = {
        'diode': comp.diode(),
        'ideal_diode': comp.diode(),
        'nonideal_diode': comp.diode(),
        'shunt': comp.resistor_parallel(),
        'series': comp.resistor_series(),
        'lgt_gen_cur': comp.current_source(),
        # 'reverse_doide': comp.reverse_doide,
        # 'transistor': comp.transistor_V,
    }

    component = ['ideal_diode', 'nonideal_diode',
                 'shunt', 'series', 'lgt_gen_cur']

    # if the terminal voltage is greater than junction voltage
    # the answer is yes
    Vj_g_Vt = True


class dark_JV(_general_JV):
    comps_avail = {
        'diode': comp.diode(),
        'ideal_diode': comp.diode(),
        'nonideal_diode': comp.diode(),
        'shunt': comp.resistor_parallel(),
        'series': comp.resistor_series(),
        'diode_res': comp.diode_resistor(),
        # 'lgt_gen_cur': comp.current_source(),
        # 'reverse_doide': comp.reverse_doide,
        # 'transistor': comp.transistor_V,
    }

    component = ['ideal_diode', 'nonideal_diode',
                 'shunt', 'series']

    # if the terminal voltage is greater than junction voltage
    # the answer is no
    Vj_g_Vt = False


class sudo_JV(_general_JV):
    '''
    this is for things like Sun Voc
    or suns PL, where there is no current
    extracted and hence no Rs
    '''
    comps_avail = {
        'diode': comp.diode(),
        'ideal_diode': comp.diode(),
        'nonideal_diode': comp.diode(),
        'shunt': comp.resistor_parallel(),
        'lgt_gen_cur': comp.current_source(),
        'diode_res': comp.diode_resistor(),
    }

    component = ['ideal_diode', 'nonideal_diode',
                 'shunt', 'lgt_gen_cur']

    # if the terminal voltage is greater than junction voltage
    # the answer is in general yes, but there is 0 difference.
    # However, for metalized contact where a shocky diode is formed
    # the Vt can differ from the Vj.
    Vj_g_Vt = True


class fitting_methods():
    '''
    A class with different types of fitting.
    None are particular special nor amazing
    '''

    default_method = 'least squares_b'

    def __init__(self, model):

        self.Vj_g_Vt = model.Vj_g_Vt
        self.model = model

        self.fit_method_dic = {
            'least squares_b': self.least_squares,
            'least squares': self.linalg,
            'other': self.scipy_minimise,
        }

    def fit(self, J, V, method):

        method = method or self.default_method
        # try:
        fitting = self.fit_method_dic[method](J, V, self.model.comps_used)
        # except:
        #     print 'fit failed, who knows why'
        #     print 'returning starting values'
        #     fitting = self.model.comps_used
        #     print fitting

        return fitting

    def linalg(self, J, V, comps_used=None):
        '''
        Tries a fit base on simple linear algebra
        There is no bounds so values can do what ever
        they want

        this is not working, as it is not passing the
        variables back in a nice form
        '''
        if comps_used is None:
            comps_used = self.model.comps_used

        if 'series' in comps_used.keys():
            comps_used = self.Rs_via_linalg(J, V, comps_used)

        else:
            comps_used = self.linalg_Vj(J, V, comps_used)

        vals = {i: k.params for i, k in comps_used.items()}

        return vals

    def Rs_via_linalg(self, J, V, comps_used):
        '''
        This uses a searching method to solve for Rs,
        which all the other values solved via

        '''
        # do we take the defaults?

        # check series is actually in the components
        assert 'series' in comps_used.keys()

        comps = comps_used.copy()
        del comps['series']

        ans = opt.minimize_scalar(self._rs_min_func,
                                  method='Brent',
                                  bounds=(0, 10),
                                  args=(J, V, comps)
                                  )

        # fit_val = np.append(self.temp, ans)

        for i, k in comps.items():
            comps_used[i] = k
        comps_used['series'].params['R'] = ans['x']

        return comps_used

    def _test_convergence(self, fun, x, *args):

        for i in x:
            y = fun(i, *args)

            plt.figure('res')
            plt.plot(abs(i), y, '.')
            plt.loglog()

    def _rs_min_func(self, R, J, V, comps_used):

        J_sim = np.zeros(V.shape)

        # determine junction voltage
        Vj = V + (-1)**(self.Vj_g_Vt) * J * abs(R)

        # fit inside components to it
        comps_used = self.linalg_Vj(J, Vj, comps_used)
        # print R,  comps_used.keys(), self.temp

        # find the current at each voltage
        for component, handle in comps_used.items():
            if 'series' != component:
                J_sim += handle.current(Vj)

        return np.sum(np.abs(J_sim - J))

    def linalg_Vj(self, J, Vj, comps):
        '''
        solves the R2 norm to find the "best" answer

        input:
            Vj: (array size M)
                this is the junction voltage, it does not include
                the "series" resistance free component
            J: (array size M)
                the terminal current density
            comps_used: (dic)
                a dictionary of the compoents to their class
            ret_dic: (bool, optional)
                if a dict of components to value should be returnedb
        '''
        # make a copy
        ccomps = comps.copy()

        # make sure the components does not include series
        try:
            del ccomps['series']
        except:
            pass

        # build coef matrix from voltage
        A = []
        for component, handle in ccomps.items():
            try:
                A.append(handle._lin_sys_comp(Vj))
            except:
                print component, 'did not have a lin_sys_comp'
                pass

        A = np.matrix(A).T

        fit_vals = nalg.lstsq(A, J)

        vals = iter(fit_vals[0])

        for handle in ccomps.values():
            val = vals.next()
            handle._set_lin_params(val)

        return ccomps

    def scipy_minimise(self, J, V, comps_used):
        '''
        This does something, but if it works
        how really knows
        '''

        x0 = []
        bounds = []

        if 'series' in comps_used.keys():
            x0.append(comps_used['series'].params['R'])
            bounds.append((0, None))

        for component, handle in comps_used.items():
            if 'series' not in component:
                for param in handle.fitting_params:
                    x0.append(handle.params[param])
                    bounds.append((0, None))

        bounds = tuple(bounds)
        x0 = np.array(x0)

        fitting = opt.minimize(
            self.scipy_minimise_fun, x0, args=(J, V, comps_used),
            bounds=bounds, method='L-BFGS-B',
        )

        if not fitting['success']:
            print 'Fitting Failed!!!!'
            print fitting

        vals = {i: k.params for i, k in comps_used.items()}

        return vals

    def scipy_minimise_fun(self, x0, J, V, comps_used):

        J_sim = np.zeros(V.shape)

        vals = iter(x0)

        Vj = _terminal_to_junction_voltage(V, J, comps_used, self.Vj_g_Vt)

        for component, handle in comps_used.items():
            if 'series' not in component:
                for param in handle.fitting_params:
                    handle.params[param] = abs(vals.next())
                    J_sim += handle.current(Vj)

        return np.sum(np.abs(J_sim - J) / J)

    def least_squares(self, J, V, comps_used=None):
        '''
        performs a least squares minimization to J(V) data, using the
        components passed.

        inputs:
            V: (array, shape M)
                terminal voltage
            I: (array, shape M)
                terminal current
            comps_used: (dic)
                a dictionary of component names, and component handles.

        returns
               vals: (dic)
                fitting values returned in a dictionary
                in the form that simulated JV classes take (comp_vals)

        '''
        x0 = []
        bounds = []
        if comps_used is None:
            comps_used = self.model.comps_used

        # a hack to pass into the fitting function
        self.comps_used = comps_used
        self.J = J

        if 'series' in comps_used.keys():
            x0.append(comps_used['series'].params['R'])

        for component, handle in comps_used.items():

            if 'series' not in component:
                for param in handle.fitting_params:
                    x0.append(abs(handle.params[param]))

        # the bounds, we want positive numbers
        bounds = (0, np.inf)

        # create the weights of the fit
        # and make sure they are not zero
        sigma = np.copy(J)
        sigma[sigma == 0] = np.amin(sigma) / 10

        # the values of fitting are not used
        # as the values are currently stored in the
        # component class
        fitting = opt.curve_fit(
            self.least_squares_fun, V, J,
            p0=(x0),
            bounds=bounds,
            # x_scale='jac',
            sigma=sigma,
            absolute_sigma=False,
        )

        vals = {i: k.params for i, k in self.comps_used.items()}
        return vals

    def least_squares_fun(self, V, *params):
        '''
        a least squares function that 
        first subtracts the voltage drop (using the actual current)
        and then does all the other components
        '''
        J_sim = np.zeros(V.shape)

        vals = iter(params)

        # if series is a parameter, determine its impact
        if 'series' in self.comps_used.keys():
            r = vals.next()
            self.comps_used['series'].params['R'] = r

        Vj = _terminal_to_junction_voltage(
            V, self.J, self.comps_used, self.Vj_g_Vt)

        for component, handle in self.comps_used.items():
            if 'series' != component:
                for param in handle.fitting_params:
                    handle.params[param] = (vals.next())

                    J_sim += handle.current(Vj)

        return J_sim
