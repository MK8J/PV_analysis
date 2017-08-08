
import numpy as np
import scipy.constants as const
import matplotlib.pylab as plt


def J0(nxc, tau, thickness, ni, method, ret_all=False, res=False,  ** kwargs):
    '''
    Caculates Jo from the measurement from a lifetime measurement
    inputs:
        nxc: (numpy array)
            number of excess carriers (cm^-3)
        tau: (numpy array)
            lifetime (s)
        thickness: (float)
            sample thickness (cm)
        ni: (float or numpy array)
            the intrinsic carrier density, or effective intrinsic carrier
            density
        method: (str)
            one of the implimented methods
        kwargs: (optional)
            additional arguments required by some methods
    '''

    method_dic = {
        'kane&swanson': _J0_KaneSwanson,
        'king': _J0_King,
        'Blum_BGN': _J0_Kimmerle,
        'Kimmerle_SRH': _J0_Kimmerle_SRH,
        'Kimmerle_Diffusion': _J0_Kimmerle_Diffusion,
    }

    J0, nxc_cor, itau, residuals = method_dic[
        method](nxc, tau, thickness, ni, **kwargs)

    # print(J0, nxc_cor, itau)

    if ret_all:
        vals = (J0, nxc_cor, itau, residuals)
    elif res:
        vals = (J0, residuals)
    else:
        vals = (J0)
    return vals


def _J0_Kimmerle_Diffusion(
        nxc, tau, thickness, ni, tau_aug, Ndop, D_ambi, ni_eff, **kwargs):
    # initialise the values
    _J0, nxc_corr, itau, residuals = _J0_Kimmerle(
        nxc, tau, thickness, ni, tau_aug, ni_eff)

    for i in range(10):
        tau_SRH = 1. / (1. / tau - 1. / tau_aug -
                        1. / (
                            const.e * thickness * ni_eff**2 / (
                                _J0 * (nxc + Ndop)
                            ) +
                            thickness**2 / D_ambi / np.pi**2)
                        )

        tau_SRH = np.mean(tau_SRH)

        if tau_SRH < 0:
            tau_cor = 1. / \
                (1. / tau - 1. / tau_aug)
            _J0, nxc_corr, itau, residuals = _J0_KaneSwanson(
                nxc, (tau_cor - thickness**2 / D_ambi / np.pi**2) / ni_eff**2, thickness, 1)
        else:
            tau_cor = 1.  / \
                (1. / tau - 1. / tau_aug - 1. / tau_SRH)
            _J0, nxc_corr, itau, residuals = _J0_KaneSwanson(
                nxc, (tau_cor - thickness**2 / D_ambi / np.pi**2) / ni_eff**2, thickness, 1)

    return _J0, nxc_corr, itau / ni**2, residuals / ni**2


def _J0_Kimmerle_SRH(nxc, tau, thickness, ni, tau_aug, Ndop, ni_eff, **kwargs):
    # initialise the values
    _J0, nxc_corr, itau, residuals = _J0_Kimmerle(
        nxc, tau, thickness, ni, tau_aug, ni_eff)
    for i in range(10):
        tau_SRH = 1. / (1. / tau - 1. / tau_aug -
                        _J0 * (nxc + Ndop) /
                        (const.e * thickness * ni_eff**2)
                        )

        tau_SRH = np.mean(tau_SRH)
        # print('\t', tau_SRH)

        if tau_SRH < 0:
            tau_cor = 1. / ni_eff**2 / (1. / tau - 1. / tau_aug)
            _J0, nxc_corr, itau, residuals = _J0_KaneSwanson(
                nxc, tau_cor, thickness, 1)
        else:
            tau_cor = 1. / ni_eff**2 / (1. / tau - 1. / tau_aug - 1. / tau_SRH)
            _J0, nxc_corr, itau, residuals = _J0_KaneSwanson(
                nxc, tau_cor, thickness, 1)

    return _J0, nxc_corr, itau / ni**2, residuals / ni**2


def _J0_Kimmerle(nxc, tau, thickness, ni, tau_aug, ni_eff, **kwargs):
    tau_cor = 1. / ni_eff**2 / (1. / tau - 1. / tau_aug)
    _J0, nxc_corr, itau, residuals = _J0_KaneSwanson(
        nxc, tau_cor, thickness, 1)

    return _J0, nxc_corr, itau / ni**2, residuals / ni**2


def _J0_King(nxc, tau, thickness, ni, tau_aug,  **kwargs):
    tau_cor = 1. / (1. / tau - 1. / tau_aug)
    return _J0_KaneSwanson(nxc, tau_cor, thickness, ni)


def _J0_KaneSwanson(nxc, tau, thickness, ni, **kwargs):

    (slope, inter), residuals, rank, singular_values, rcond = np.polyfit(
        nxc, 1. / tau, 1, full=True)

    return const.e * thickness * ni**2 * slope, nxc, 1. / tau, np.sqrt(residuals)
