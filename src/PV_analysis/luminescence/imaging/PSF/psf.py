import numpy as np
import matplotlib.pylab as plt
import scipy.interpolate
import scipy.fftpack as fft
from scipy.signal import savgol_filter

from .PSF_from_ESF_teal import teals_method


def load_test_ESF():
    '''
    returns a measured ESF
    '''
    fname = r'ESF.dat'
    data = np.genfromtxt(fname, names=True)
    data['pixel'] = data['pixel']  # [::-1]
    return data


def PSF_2D_from_1D(PSF_line, extend=False, right=None, psf_length=1024):
    '''
    Generates a 2D PSF from a line
    Basically rotates a line to form an image
    '''

    if extend:
        adjust = np.ones(
            psf_length - PSF_line.shape[0]) * np.amin(PSF_line)
        _PSF_line = np.append(PSF_line, adjust)
    else:
        _PSF_line = np.copy(PSF_line)

    def f(x, y):

        r = np.sqrt(
            (x - _PSF_line.shape[0] + 1)**2 + (y - _PSF_line.shape[0] + 1)**2)
        if right is None:
            val = np.interp(r, range(_PSF_line.shape[0]), _PSF_line)
        else:
            val = np.interp(
                r, range(_PSF_line.shape[0]), _PSF_line, right=right)
        return val

    # the 2D matrix needs to be an odd number, such that is can have a
    # central peak
    PSF = np.fromfunction(f,
                          (_PSF_line.shape[0] * 2 - 1,
                           _PSF_line.shape[0] * 2 - 1),
                          dtype=float)

    return PSF


def LSF_from_2D_PSF(PSF, method='sum'):
    method_dic = {
        'sum': _LSF_from_2D_PSF_sum,
        'FFT': _LSF_from_2D_PSF_FFT
    }

    if method not in method_dic.keys():
        print('provided method not found try one of:')
        for key in method_dic.keys():
            print(key)
        raise

    return method_dic[method](PSF)


def _LSF_from_2D_PSF_FFT(PSF):
    '''
    Determined a LSF from a PSF via convolution
    with a line
    '''
    Line = np.zeros(PSF.shape)
    Line[Line.shape[0] / 2, :] = 1
    LSF_2D = np.fft.fftshift(
        np.fft.ifft2(np.fft.fft2(PSF) * np.fft.fft2(Line)))

    index = np.argmax(LSF_2D[:, LSF_2D.shape[0] / 2])
    return LSF_2D[index:, 0]


def _LSF_from_2D_PSF_sum(PSF):
    '''
    Determined a LSF from the sum in the vertical direction of a 2D symetric PSF
    '''
    LSF = np.sum(PSF, axis=1)
    index = np.argmax(LSF)

    # this is to compare if the LSf each way is the same
    # and the same legnth. That way we know the PSF is centered

    # in the 2D matrix
    # plt.figure('LSF')
    # plt.plot(LSF[index:], '.-')
    # plt.plot(LSF[:index + 1][::-1], '.')
    # plt.semilogy()
    assert LSF[index:].shape == LSF[:index + 1].shape

    return LSF[index:]


def LSF_from_ESF(data, method='poly', ignore_index=None, order=10):
    method_dic = {
        'poly': _LSF_from_data_polyfit,
        'spline': _LSF_from_data_spline_peicemeal,
        'num_diff': _LSF_from_data_diff,
    }

    if method not in method_dic.keys():
        print('provided method not found try one of:')
        for key in method_dic.keys():
            print(key)
        raise

    return method_dic[method](data=data, ignore_index=ignore_index, order=order)


def _LSF_from_data_polyfit(data, ignore_index=None, order=10):
    '''
    determine a LSF from a polynomial fit
    to the log log of an edge spread function
    '''

    if ignore_index is None:
        ignore_index = 0

    # remove the data points to be removed
    data_fit = np.copy(data[ignore_index:])

    data_fit['pixel'] += 1 - data_fit['pixel'][0]
    # create the output
    ESF = np.copy(data['counts'])

    p0 = np.polyfit(
        np.log(data_fit['pixel']), np.log(data_fit['counts']), order,
        w=1. / np.sqrt(data_fit['pixel'])
    )

    ESF[ignore_index:] = np.exp(np.polyval(p0, np.log(data_fit['pixel'])))

    # plt.loglog()
    # finally the values
    # LSF = abs(np.diff(ESF))
    # a better diff
    LSF = abs(np.gradient(ESF))

    return LSF


def _LSF_from_data_spline_peicemeal(data, ignore_index=None, order=5):
    '''
    determines the LSF from a spline fit to the ESF.
    inputs is the ESF
    '''

    if ignore_index is None:
        ignore_index = 0

    if order > 5:
        print('Spline order can not be >5, order set to 5')
        order = 5

    data_fit = np.copy(data[ignore_index:])
    ESF = np.copy(data['counts'])

    spl = scipy.interpolate.UnivariateSpline(
        np.log(data_fit['pixel']), np.log(data_fit['counts']), k=order, s=10,
        w=1. / np.sqrt(data_fit['pixel']))

    ESF[ignore_index:] = np.exp(spl(np.log(data_fit['pixel'])))

    LSF = abs(np.diff(ESF))

    return LSF


def _LSF_from_data_diff(data, **args):
    '''
    LSF directly from numerical differentiation of data
    '''
    LSF = abs(np.diff(data['counts']))

    return LSF


def _norm_2D_PSF(psf):

    assert psf.ndim == 2
    return psf / np.sum(psf)


def PSF_from_LSF(LSF, method='teal', **args):
    '''
    Determines a PSF from an ESF

    inputs:
        LSF: (1D array)
            The line spread function
        method: (string)
            The method to use to caculate the PSF
        **agrs: (others)
            Other values to be passed to the different methods
    '''
    method_dic = {
        'otto': _PSF_with_Ottos_method,
        'teal': teals_method,
        'old_skool': _PSF_from_FFT_LSF_scipy_doublesided,
    }

    if method not in method_dic.keys():
        print('provided method not found try one of:')
        for key in method_dic.keys():
            print(key)
        raise

    return method_dic[method](LSF, **args)


def _PSF_with_Ottos_method(LSF, iterations=10, psf_guess=None):
    '''
    determines the PSF of an image from the LSF.
    It is an itterative method that adjust the PSF until
    when convoled with a line provides the LSF
    '''
    LSF1 = np.copy(LSF)
    LSF1 /= np.amax(LSF1)

    if psf_guess is None:
        psf_guess = np.copy(LSF1)

    for i in range(1):

        PS = PSF_2D_from_1D(np.abs(psf_guess))
        err = LSF1 / LSF_from_2D_PSF(PS)
        psf_guess *= abs(err) / 2

    for i in range(iterations):
        # print i

        PS = PSF_2D_from_1D(psf_guess)
        err = LSF1 / LSF_from_2D_PSF(PS)

        psf_guess *= 1 - (1 - abs(err)) / 2

    PS = PSF_2D_from_1D(psf_guess)
    PS = _norm_2D_PSF(PS)

    center = PS.shape[0] / 2

    return PS[center, center:]


def _PSF_from_FFT_LSF_numpy(LSF):
    '''
    This is an old method from the 50's.
    It says the Fourier transform of the PSF is
    the Fourier transform  of the LSF rotated around the
    y axis:
        F(PSF)[r,theta] = int F(LSF)[r]

    This uses the numpy implementation of FFT, and rFFT.
    '''
    # See the comments for the scipy implementation
    LSF1 = np.copy(LSF)
    LSF1 /= np.amax(LSF1)

    F_LSF = np.fft.rfft(LSF1)
    F_PSF = np.zeros((F_LSF.shape[0] * 2, F_LSF.shape[0] * 2))

    center = F_LSF.shape[0]
    x = np.arange(0, F_LSF.shape[0], 1)
    i = np.arange(0, F_PSF.shape[1], 1)
    for j in range(F_PSF.shape[1]):
        index = np.sqrt((i - center)**2 + (j - center)**2)

        real = np.interp(
            index, x, np.real(F_LSF), right=np.real(F_LSF[-1]))
        img = np.interp(
            index, x, np.imag(F_LSF), right=np.imag(F_LSF[-1]))
        val = real + 1j * img

        F_PSF[:, j] = val

    y = np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(F_PSF)))[center, center:]

    return abs(y)


def _PSF_from_FFT_LSF_scipy_doublesided(LSF, center_shift=0.5,
                                        LSF_splice_start=0,
                                        LSF_splice_end=-1):
    '''
    This is an old method from the 50's.
    It says the Fourier transform of the PSF is
    the Fourier transform  of the LSF rotated around the
    y axis:
        F(PSF)[r,theta] = int F(LSF)[r]

    the idea of the function is to use a entire LSF function
    to get a longer PSF. It seems to agree with Ottos method, but requires
    more testing to be sure.

    inputs:
        LSF : (array)
            the Line spread function
        center_shift: (float)
            A shift in the centre index
        LSF_splice_start: (float)
            The index for the LSF splice to start,
            i.e. should it include the first point twice
        LSF_splice_end: (float)
            The index for the LSF splice to start,
            i.e. should it include the last point twice
    '''
    LSF1 = np.copy(LSF)
    LSF1 /= np.amax(LSF1)
    print(LSF.shape)

    # join it with its self
    LSF1 = np.hstack((LSF1, LSF1[LSF_splice_start:LSF_splice_end][::-1]))
    # plt.figure('F_LSF')
    # plt.plot(LSF1)
    # plt.semilogy()
    # FFT it to find the PSF
    F_LSF0 = fft.fft(LSF1, n=LSF1.shape[0])
    F_PSF = np.zeros((F_LSF0.shape[0], F_LSF0.shape[0]))

    # define a centre and, the RHS of the LSF
    # the result is very sensitive to this
    center = int(F_LSF0.shape[0] / 2.)
    center_index = center
    F_LSF = F_LSF0[center_index:][::-1]
    # plt.plot(F_LSF0, '.-')
    # plt.plot(F_LSF0[center - 1:][::-1], '.')

    # create arrays for interpolation
    x = np.arange(0, F_LSF.shape[0], 1)
    # array used for 1st dim on F_PSF
    i = np.arange(0, F_PSF.shape[1], 1)
    # rotate the F_LSF around the second axis
    for j in range(F_PSF.shape[1]):

        # convert i j into r
        r = np.sqrt((i - center_index + 1)**2 + (j - center_index + 1)**2)

        # get real and image at that r value
        real = np.interp(
            r, x, np.real(F_LSF),
            # right=np.real(F_LSF[-1]))
            # check what happens if the end is 0 and not the last value
            # the results look worse. But really shows that more data is
            # needed.
            right=0)

        # checked both setting to last value and to 0.
        # setting to 0 wins
        img = np.interp(
            r, x, np.imag(F_LSF),
            # right=np.imag(F_LSF[-1]))
            right=0)

        # write to array
        F_PSF[:, j] = real + 1j * img

    # get iFFt i.e the PSF
    PSF = fft.ifft2(F_PSF, shape=(F_PSF.shape[0], F_PSF.shape[0]))
    # get the components at the right place
    PSF = fft.fftshift(PSF)
    # normalise it
    PSF = _norm_2D_PSF(abs(PSF))
    # an abs to remove the tiny imaginary component that should not exist
    # PSF.imag = 0
    # print(center_index, 'old')
    # center_index = np.argmax(PSF[center_index, :])
    PSF = PSF[center_index, center_index:]
    # print(center_index, 'yeah')
    # print(PSF.shape)
    return PSF


def _PSF_from_FFT_LSF_scipy(LSF):
    '''
    This is an old method from the 50's.
    It says the Fourier transform of the PSF is
    the Fourier transform  of the LSF rotated around the
    y axis:
        F(PSF)[r,theta] = int F(LSF)[r]

    This uses the scipy implementation of FFT
    '''
    # create the FFT
    F_LSF = fft.fft(LSF, n=LSF.shape[0])
    F_PSF = np.zeros((F_LSF.shape[0], F_LSF.shape[0]))

    # define a center and, the RHS of the LSF
    center = (F_LSF.shape[0] - 1) / 2
    F_LSF = F_LSF[:center]

    # create arrays for interpolation
    x = np.arange(0, F_LSF.shape[0], 1)
    # array used for 1st dim on F_PSF
    i = np.arange(0, F_PSF.shape[1], 1)

    # rotate the F_LSF around the second axis
    for j in range(F_PSF.shape[1]):

        # convert i j into r
        index = np.sqrt((i - center)**2 + (j - center)**2)
        # get real and image at that r value
        real = np.interp(
            index, x, np.real(F_LSF),
            right=np.real(F_LSF[-1]))
        # check what happens if the end is 0 and not the last value
        # the results look worse. But really shows that more data is needed.
        # right=0)

        img = np.interp(
            index, x, np.imag(F_LSF),
            right=np.imag(F_LSF[-1]))
        # right=0)

        # write to array
        F_PSF[:, j] = real + 1j * img

    # get iFFt i.e the PSF
    PSF = fft.ifft2(F_PSF, shape=(F_PSF.shape[0], F_PSF.shape[0]))
    # get linescan of PSF, shift so peak is in the middle

    center = PSF.shape[0] / 2.
    PSF = abs(fft.fftshift(PSF))

    # an abs to remove the tiny imaginary component that should not exist
    return PSF[center, center:]


def _plot_FT_norm(values, axes, norm=False):
    '''
    a private function that plots the FT of a function
    '''
    if norm:
        norm = np.amax(np.abs(np.fft.fft(values)))
    else:
        norm = 1.

    axes.plot(np.fft.fftfreq(values.shape[0]),
              np.abs(np.fft.fft(values)) / norm)


def _funct(x, alpha):
    return np.exp(-alpha * x) * (alpha * x + 1) / x**2


def plot_function(x):
    _funct(x, 64) - funct(x, 1e-6)


def TEST_PSF_Derivation():
    # load datas
    datas = load_test_ESF()
    data = datas[23:]

    index = data['pixel'] < 120
    index *= data['pixel'] < 140

    data = data[index]

    # differen LSF's
    LSF_0 = LSF_from_ESF(data, method='num_diff')
    # datam = savgol_filter(data, 51, 11)
    # LSF_01 = np.gradient(datam)

    LSF_1 = LSF_from_ESF(data, method='poly', ignore_index=5, order=5)
    LSF_2 = LSF_from_ESF(data, method='spline', ignore_index=5, order=5)

    fig, ax = plt.subplots(2, 3, figsize=(18, 6))
    fig_temp, ax_temp = plt.subplots(1, figsize=(8, 6))

    for axes in ax.flatten():
        axes.set_color_cycle(['b', 'g', 'r'])

    # plot of ESF raw data
    ax_temp.plot(data['counts'] / data['counts'][0], 'k.-')

    # plot the FT of the ESF
    _plot_FT_norm(data['counts'], ax[1][0], norm=True)

    # plot the LSF
    ax[0][1].plot(LSF_0 / np.amax(LSF_0), '.', label='Direct Derivative')
    ax[0][1].plot(
        LSF_1 / np.amax(LSF_1),
        '.',
        label='poly_fit 10 (ignore first 5 points)')
    ax[0][1].plot(
        LSF_2 / np.amax(LSF_2),
        '.',
        label='poly fit 5 (ignore first 5 points)')

    _plot_FT_norm(LSF_0, ax[1][0], norm=True)
    _plot_FT_norm(LSF_1, ax[1][0], norm=True)
    _plot_FT_norm(LSF_2, ax[1][0], norm=True)

    for LSF in [LSF_1, LSF_2]:
        # print(LSF.shape)
        PSo = PSF_from_LSF(LSF, method='otto', iterations=30)
        PS = PSF_from_LSF(LSF, method='old_skool')
        PSt = PSF_from_LSF(LSF, method='teal')
        # PS1 = PSF_from_FFT_LSF(LSF)

        ax[0][2].plot(PS / PS[0], ',-')
        # print PS
        ax[0][2].plot(PSo / PSo[0], '--')
        ax[0][2].plot(PSt / PSt[0], ':')

        # r = np.arange(0, PS.shape[0], 1)
        # area = np.pi * r**2
        # area = area[1:] - area[:-1]

        # ax[1][2].plot(PS[1:] * area)
        # ax[1][1].plot(np.cumsum(PS[1:] * area) / np.cumsum(PS[1:] *
        # area)[-1])

        # _plot_FT_norm(PS, ax[1][0], norm=True)
        # _plot_FT_norm(PSo, ax[1][0], norm=True)
        PS = PSF_2D_from_1D(PSt)
        LSf = LSF_from_2D_PSF(PS)
        # ax[0][1].plot(LSf / np.amax(LSf), '-.')

    ax[0][1].plot(np.inf, np.inf, '-.', label='LSF from PSF')
    ax[0][2].plot(np.inf, np.inf, '--', label='Otts method')
    ax[0][2].plot(np.inf, np.inf, ',-', label='50\'s method')

    ax_temp.plot(np.cumsum(LSF_0[::-1])[::-1] / np.cumsum(LSF_0[::-1])
                 [-1] + data['counts'][-1] / data['counts'][0] - LSF_0[-1],
                 '--')
    ax_temp.plot(np.cumsum(LSF_1[::-1])[::-1] / np.cumsum(LSF_1[::-1])
                 [-1] + data['counts'][-1] / data['counts'][0] - LSF_1[-1],
                 '--')
    ax_temp.plot(np.cumsum(LSF_2[::-1])[::-1] / np.cumsum(LSF_2[::-1])
                 [-1] + data['counts'][-1] / data['counts'][0] - LSF_2[-1],
                 '--')

    # plt.plot(ESF, 'g:')
    ax_temp.semilogx()
    ax_temp.semilogy()
    ax[0][1].semilogy()
    ax[0][2].semilogy()

    ax_temp.set_title('ESF')
    ax[0][1].set_title('LSF')
    ax[0][2].set_title('PSF')

    ax[0][1].legend(loc=0)
    ax[0][2].legend(loc=0)
    ax[0][2].set_ylim(bottom=1e-6)
    # ax[1][0].set_ylim(bottom=1e-6)

    # titles

    ax[0][1].set_title('LSF')
    ax[0][2].set_title('PSF determined')
    ax[1][0].set_title('MTF')
    ax[1][1].set_title('Col sum of PSF')
    ax[1][2].set_title('Photons spreading to distance')

    ax[1][0].loglog()
    ax[1][2].semilogy()
    plt.show()


if __name__ == "__main__":
    TEST_PSF_Derivation()
