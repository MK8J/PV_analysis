

import numpy as np
from semiconductor.electrical.mobility import Mobility


def derivative(nxc, I_pl, f_cutoff, t_step):
    '''
    Uses the relationship between PL and the number of excess carriers to find the
    bulk doping of a sample.
    '''
    # Make the Fourier good
    APL_fft = fft.rfft(I_pl)
    APC_fft = fft.rfft(nxc)
    F_fft = fft.rfftfreq(I_pl.shape[0], t_step)

    # Remove higher frequencies, which can not be in
    # the signal
    APL_fft[F_fft > f_cutoff] = 0
    APC_fft[F_fft > f_cutoff] = 0

    # converting into real world
    I_pl = fft.irfft(APL_fft)
    nxc = fft.irfft(APC_fft)

    # Now for differentiation
    timediiff = np.ones(data['PL'].shape)*(t_step)

    dPL = np.gradient(I_pl, t_step)
    dnxc = np.gradient(nxc, t_step)


    cof = np.polyfit(nxc, (dPL/dnxc), 1)

    return cof[1]/cof[0]*2)
