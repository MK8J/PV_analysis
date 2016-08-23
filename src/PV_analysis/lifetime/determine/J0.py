
import numpy as np
import scipy.constants as const
import matplotlib.pylab as plt


def J0(nxc, tau, thickness, ni, method, **kwargs):
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
        'Kimmerle_BGN': _J0_Kimmerle,
        'Kimmerle_SRH': _J0_Kimmerle_SRH,
        'Kimmerle_Diffusion': _J0_Kimmerle_Diffusion,
    }

    # for some reason I was getting an array back for some of the fits,
    # so this
    # print(kwargs, method)

    print('\n\n\n')
    # print(method_dic[method](nxc, tau, thickness, ni, **kwargs))
    # print(nxc, tau, thickness, ni)
    print('\n\n\n')
    return np.asarray([method_dic[method](nxc, tau, thickness, ni, **kwargs)]).flatten()[0]


def _J0_Kimmerle_Diffusion(
        nxc, tau, thickness, ni, tau_aug, Ndop, D_ambi, **kwargs):
    # initialise the values
    _J0 = _J0_Kimmerle(nxc, tau, thickness, ni, tau_aug)

    for i in range(10):
        tau_SRH = 1. / (1. / tau - 1. / tau_aug -
                        1. / (
                            const.e * thickness * ni**2 / (
                                2 * _J0 * (nxc + Ndop)
                            ) +
                            thickness**2 / D_ambi / np.pi**2)
                        )

        tau_SRH = np.mean(tau_SRH)

        tau_cor = 1. / ni**2 / (1. / tau - 1. / tau_aug - 1. / tau_SRH)
        _J0 = _J0_KaneSwanson(nxc, tau_cor, thickness, 1)

    return _J0


def _J0_Kimmerle_SRH(nxc, tau, thickness, ni, tau_aug, Ndop, **kwargs):
    # initialise the values
    _J0 = _J0_Kimmerle(nxc, tau, thickness, ni, tau_aug)
    for i in range(10):
        tau_SRH = 1. / (1. / tau - 1. / tau_aug -
                        2 * _J0 * (nxc + Ndop) /
                        (const.e * thickness * ni**2)
                        )

        tau_SRH = np.mean(tau_SRH)

        tau_cor = 1. / ni**2 / (1. / tau - 1. / tau_aug - 1. / tau_SRH)
        _J0 = _J0_KaneSwanson(nxc, tau_cor, thickness, 1)
    return _J0


def _J0_Kimmerle(nxc, tau, thickness, ni, tau_aug, **kwargs):
    tau_cor = 1. / ni**2 / (1. / tau - 1. / tau_aug)
    return _J0_KaneSwanson(nxc, tau_cor, thickness, 1)


def _J0_King(nxc, tau, thickness, ni, tau_aug, **kwargs):
    tau_cor = 1. / (1. / tau - 1. / tau_aug)
    return _J0_KaneSwanson(nxc, tau_cor, thickness, ni)


def _J0_KaneSwanson(nxc, tau, thickness, ni, **kwargs):
    slope, inter = np.polyfit(nxc, 1. / tau, 1)

    return const.e * thickness * ni**2 * slope
