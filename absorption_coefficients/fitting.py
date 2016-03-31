
import numpy as np


def Marfarlane_fit(E, alpha_BB, power, lower, upper, mult_en=False):
    '''
    performs a line fit to alpha data, 
    where the alpha(E^power). This is based 
    on the work of:

    G. Macfarlane, T. McLean, J. Quarrington, and V. Roberts, Phys. Rev. 111, 1245 (1958).
    DOI:10.1103/PhysRev.111.1245

    inputs:

        E: (numpy array of floats)
            the photon energy
        alpha_BB: (numpy array of floats)
            the band-to-band absorption coefficeints
        power: (float)
            the power relationship between the energy and absorption coefficient
            for indriect is 2, for direct is 0.5, for exciton is 0.5-1.5
        lower: (float)
            the lower limit in energy for the linear fitting
        upper: (float)
            the upper limit in energy for the linear fitting

    output:
        (array, energy gap)
        the output alpha values
        and the energy gap
    '''
    if mult_en:
        alpha_BB *= E

    # Find where E < value, and E > value2
    index = E > lower
    index *= E < upper

    # we got out alphha ^power
    # may want to times alpha here by E
    y = np.power(alpha_BB, 1. / power)[index]

    # we got energies
    x = E[index]

    # fit it
    p = np.polyfit(x, y, 1)

    # determine the energy gap
    energy_gap = -p[1] / p[0]

    # create the fitted alpha
    alpha = np.polyval(p, E)

    # remove impossible negitive fitting values
    alpha[alpha < 0] = 0

    # turn back into alpha
    alpha = np.power(alpha, power)

    # get alpha back to the way it was
    if mult_en:
        alpha_BB /= E

    # returns the fitted alpha, and energy gap
    return alpha, energy_gap
