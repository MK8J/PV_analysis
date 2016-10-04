
from semiconductor.recombination.intrinsic import Intrinsic
from semiconductor.recombination.extrinsic import SRH
import scipy.optimize as opt
import matplotlib.pylab as plt


int_recom = Intrinsic(material='Si',
                      temp=300,
                      ni_author=None,
                      rad_author=None,
                      aug_author=None,
                      Nd=0,
                      Na=1e16,
                      )

srh = SRH(
    material='Si',
    defect=None,
    temp=300.,
    vth_author=None,
    Nt=1e10,  # the number of traps
    Nd=0,
    Na=1e16,
    nxc=1e10,
    ni_author=None
)


def general_fit(nxc, tau, Na, Nd,
                held_Et=None, held_sigma_ratio=None, hold_number=None):

    var = {
        'Na': Na,
        'Nd': Nd,
    }

    int_recom.caculationdetails = var
    srh.caculationdetails = var

    itau = 1. / tau - int_recom.itau(nxc)

    def _fitting(nxc, Et, sigma_ratio, number):

            et = held_Et or Et
            ratio = held_sigma_ratio or sigma_ratio
            nt = hold_number or number

            srh.usr_vals(Et=et, sigma_e=1, sigma_h=ratio, Nt=nt)
            itau = srh.itau()

            return 1. / itau

    pop, conv = opt.curve_fit(_LS_fitting, nxc, tau)
    return pop


def _LS_fitting(nxc, Et, sigma_e, sigma_h):
    '''
    Just a function that cals the lifetime, and returns it
    for a sample with a single SRH defect

    This assumes Delta n = Delta p
    '''

    Et = 0
    # srh.usr_vals(Et=Et, sigma_e=sigma_e, sigma_h=sigma_h, Nt=1.)
    srh.usr_vals(Et=Et, sigma_e=sigma_e, sigma_h=sigma_h, Nt=1.)
    itau = srh.itau(nxc=nxc)

    if sigma_e < 0 or sigma_h < 0:
        itau *= 100

    itau[itau < 0] = 0

    return 1. / itau


def fit_SRH(nxc, tau, Na, Nd):
    '''
    Fits a single Shockley Reah Hall defect to lifetime data
    The lifetime data is automatically Auger corrected.

    inputs:
        nxc: (array like)
            the number of excess carriers
        tau: (array like)
            the measured effective lifetime
        Na: (float)
            The number of acceptor atoms
        Nd: (float)
            The number of donor atoms

    returns:
        Et: (float)
            Energy level
        sigma_e: (float)
            The electron capture cross section
        sigma_h: (float)
            The hole capture cross section
    '''

    var = {
        'Na': Na,
        'Nd': Nd,
    }

    int_recom.caculationdetails = var
    srh.caculationdetails = var

    itau = 1. / tau - int_recom.itau(nxc)

    # plt.plot(nxc, 1. / itau, '--')

    pop, conv = opt.curve_fit(_LS_fitting, nxc, 1./itau, p0=(0, 5e-13, 2e-17))

    return pop


def plot_fit(nxc, Na, Nd, Et, sigma_e, sigma_h):

    var = {
        'Na': Na,
        'Nd': Nd,
    }

    int_recom.caculationdetails = var
    srh.caculationdetails = var

    itau_srh = 1. / _LS_fitting(nxc, Et, sigma_e, sigma_h)

    itau_int = int_recom.itau(nxc)

    plt.plot(nxc, 1./itau_srh,':')
    # plt.plot(nxc, 1./itau_int, ':')

    return 1. / (itau_int + itau_srh)
