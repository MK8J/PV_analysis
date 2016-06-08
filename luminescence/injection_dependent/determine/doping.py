import numpy as np
import numpy.fft as fft
import matplotlib.pylab as plt

def doping(nxc, I_pl, f_cutoff, t_step, method='derivative'):
    return _derivative(nxc, I_pl, f_cutoff, t_step)

def _derivative(nxc, I_pl, f_cutoff, t_step):
    '''
    Uses the relationship between PL and the number of excess carriers to find the
    bulk doping of a sample.
    '''
    # Make the Fourier good
    Fpl_fft = fft.rfft(I_pl)
    Fnxc_fft = fft.rfft(nxc)
    F_fft = fft.rfftfreq(I_pl.shape[0], t_step)

    # Remove higher frequencies, which can not be in
    # the signal
    Fpl_fft[F_fft > f_cutoff] = 0
    Fnxc_fft[F_fft > f_cutoff] = 0

    # converting into real world
    I_pl = fft.irfft(Fpl_fft)
    nxc = fft.irfft(Fnxc_fft)

    plt.figure('FFT')
    # print (F_fft.shape[0], I_pl.shape)
    plt.plot(F_fft, abs(Fpl_fft),'.')
    plt.semilogx()

    # Now for differentiation
    # timediiff = np.ones(I_pl.shape)*(t_step)

    dPL = np.gradient(I_pl, t_step)
    dnxc = np.gradient(nxc, t_step)

    index = nxc < 0.90*np.amax(nxc)
    index *= nxc > 0.02*np.amax(nxc)



    dn = np.linspace(np.amin(nxc[index]),np.amax(nxc[index]), 100)
    weights = np.gradient(nxc[index])
    dn = dn[1:]
    # index2 = nxc[index]
    cof = np.polyfit(nxc[index], (dPL/dnxc)[index], 1, w=weights)
    # y = np.interp(dn,nxc[index])

    plt.figure('derivative')
    plt.plot(nxc, dPL/dnxc, '.', label='raw data')
    x = np.linspace(np.amin(nxc[index]),np.amax(nxc[index]), 100)
    plt.plot(dn, np.polyval(cof, dn),'--', label='fitted data')
    plt.xlabel('Excess carrier density (cm$^{-3}$)')
    plt.ylabel('dpl/dn')
    plt.legend(loc=0)

    return cof[1]/cof[0]*2
