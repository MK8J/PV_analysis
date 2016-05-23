import numpy as np
import matplotlib.pylab as plt
import scipy.interpolate
import scipy.fftpack as fft


class PSF():

    def load_test_ESF(self):
        '''
        returns a measured ESF
        '''
        fname = r'ESF.dat'
        data = np.genfromtxt(fname, names=True)
        data['pixel'] = data['pixel']  # [::-1]
        return data

    def PSF_2D_from_linescan(self, PSF_line, extend=False, right=None):
        '''
        Generates a 2D PSF from a line
        Basically rotates a line to form an image
        '''

        exit_length = 1024

        if extend:
            adjust = np.ones(
                1024 - PSF_line.shape[0]) * np.amin(PSF_line)
            _PSF_line = np.append(PSF_line, adjust)
        else:
            _PSF_line = np.copy(PSF_line)

        def f(x, y):

            r = np.sqrt(
                (x - _PSF_line.shape[0])**2 + (y - _PSF_line.shape[0])**2)
            if right is None:
                val = np.interp(r, range(_PSF_line.shape[0]), _PSF_line)
            else:
                val = np.interp(
                    r, range(_PSF_line.shape[0]), _PSF_line, right=right)
            return val

        # print(_PSF_line.shape, PSF_line.shape)
        PSF = np.fromfunction(f,
                              (_PSF_line.shape[0] * 2,
                               _PSF_line.shape[0] * 2),
                              dtype=float)

        # print(PSF.shape)

        return PSF

    def LSF_from_PSF(self, PSF):
        '''
        Determined a LSF from a PSF via convolution
        with a line
        '''
        Line = np.zeros(PSF.shape)
        Line[Line.shape[0] / 2, :] = 1
        LSP_2D = np.fft.fftshift(
            np.fft.ifft2(np.fft.fft2(PSF) * np.fft.fft2(Line)))

        return LSP_2D[LSP_2D.shape[0] / 2:, 0]

    def LSF_from_data_polyfit(self, data, index=None, order=10):
        '''
        determine a LSF from a polynomial fit
        to the log log of an edge spread function
        '''
        if index is None:
            index = 0

        # remove the data points to be removed
        data_fit = np.copy(data[index:])
        data['pixel'] -= data['pixel'][0]

        # create the output
        ESF = np.copy(data['counts'])

        p0 = np.polyfit(
            np.log(data_fit['pixel']), np.log(data_fit['counts']), order,
            w=1. / np.sqrt(data_fit['pixel']))

        ESF[index:] = np.exp(np.polyval(p0, np.log(data_fit['pixel'])))

        # finally the values
        LSF = abs(np.diff(ESF))
        # a better diff
        # LSF = abs(np.gradient(ESF)[:-1])

        return LSF

    def LSF_from_data_spline_peicemeal(self, data, index=None):
        '''
        determines the LSF from a spline fit to the ESF.
        inputs is the ESF
        '''

        if index is None:
            index = 0

        data_fit = np.copy(data[index:])
        ESF = np.copy(data['counts'])

        spl = scipy.interpolate.UnivariateSpline(
            np.log(data_fit['pixel']), np.log(data_fit['counts']), k=3, s=10,
            w=1. / np.sqrt(data_fit['pixel']))

        ESF[index:] = np.exp(spl(np.log(data_fit['pixel'])))

        LSF = abs(np.diff(ESF))

        return LSF

    def LSF_from_data_diff(self, data):
        '''
        LSF directly from numerical differentiation of data
        '''
        LSF = abs(np.diff(data['counts']))

        return LSF

    def _norm_2D_PSF(self, psf):

        assert psf.ndim == 2
        return psf / np.sum(psf)

    def PSF_from_Ottos_method(self, LSF, iterations=10, psf_guess=None):
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

            PS = self.PSF_2D_from_linescan(np.abs(psf_guess))
            err = LSF1 / self.LSF_from_PSF(PS)
            psf_guess *= abs(err) / 2

        for i in range(iterations):
            # print i

            PS = self.PSF_2D_from_linescan(psf_guess)
            err = LSF1 / self.LSF_from_PSF(PS)

            psf_guess *= 1 - (1 - abs(err)) / 2

        PS = self.PSF_2D_from_linescan(psf_guess)
        PS = self._norm_2D_PSF(PS)

        center = PS.shape[0] / 2

        return PS[center, center:]

    def PSF_from_FFT_LSF(self, LSF):
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

    def PSF_from_FFT_LSF_scipy_doublesided(self, LSF, center_shift=-0.5,
                                           LSF_splice_start=0,
                                           LSF_splice_end=-1):
        '''
        This is an old method from the 50's.
        It says the Fourier transform of the PSF is
        the Fourier transform  of the LSF rotated around the 
        y axis:
            F(PSF)[r,theta] = int F(LSF)[r]

        the idea of the class is to use a entire LSF function
        to get a longer PSF. It works and agrees 
        with Ottos very well


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

        # join it with its self
        LSF1 = np.hstack((LSF1, LSF1[LSF_splice_start:LSF_splice_end][::-1]))

        # FFT it to find the PSF
        F_LSF = fft.fft(LSF1, n=LSF1.shape[0])
        F_PSF = np.zeros((F_LSF.shape[0], F_LSF.shape[0]))

        # define a centre and, the RHS of the LSF
        # the result is very sensitive to this
        center = float(F_LSF.shape[0]) / 2. + center_shift
        F_LSF = F_LSF[:center]
        # plt.plot(F_LSF)

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
                # right=np.real(F_LSF[-1]))
                # check what happens if the end is 0 and not the last value
                # the results look worse. But really shows that more data is
                # needed.
                right=0)

            # checked both setting to last value and to 0.
            # setting to 0 wins
            img = np.interp(
                index, x, np.imag(F_LSF),
                # right=np.imag(F_LSF[-1]))
                right=0)

            # write to array
            F_PSF[:, j] = real + 1j * img

        # get iFFt i.e the PSF
        PSF = fft.ifft2(F_PSF, shape=(F_PSF.shape[0], F_PSF.shape[0]))
        # get the components at the right place
        PSF = fft.fftshift(PSF)
        # normalise it
        PSF = self._norm_2D_PSF(abs(PSF))
        # an abs to remove the tiny imaginary component that should not exist
        # PSF.imag = 0
        PSF = PSF[center, center:]

        return PSF

    def PSF_from_FFT_LSF_scipy(self, LSF):
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

    def _plot_FT_norm(self, values, axes, norm=False):
        '''
        a private function that plots the FT of a function
        '''
        if norm:
            norm = np.amax(np.abs(np.fft.fft(values)))
        else:
            norm = 1.

        axes.plot(np.fft.fftfreq(values.shape[0]),
                  np.abs(np.fft.fft(values)) / norm)

    def _funct(self, x, alpha):
        return np.exp(-alpha * x) * (alpha * x + 1) / x**2

    def plot_function(self, x):
        self._funct(x, 64) - self.funct(x, 1e-6)


def TEST_PSF_Derivation():
    # load datas
    psf = PSF()
    datas = psf.load_test_ESF()
    data = datas[23:]

    index = data['pixel'] < 120
    index *= data['pixel'] < 140

    data = data[index]

    # differen LSF's
    LSF_0 = psf.LSF_from_data_diff(data)
    LSF_1 = psf.LSF_from_data_polyfit(data, 5)
    LSF_2 = psf.LSF_from_data_spline_peicemeal(data, 5)

    fig, ax = plt.subplots(2, 3, figsize=(18, 6))
    fig_temp, ax_temp = plt.subplots(1, figsize=(8, 6))

    for axes in ax.flatten():
        axes.set_color_cycle(['b', 'g', 'r'])

    # plot of ESF raw data
    ax_temp.plot(data['counts'] / data['counts'][0], 'k.-')

    # plot the FT of the ESF
    psf._plot_FT_norm(datas['counts'], ax[1][0], norm=True)

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

    psf._plot_FT_norm(LSF_0, ax[1][0], norm=True)
    psf._plot_FT_norm(LSF_1, ax[1][0], norm=True)
    psf._plot_FT_norm(LSF_2, ax[1][0], norm=True)

    for LSF in [LSF_1, LSF_2]:
        PSo = psf.PSF_from_Ottos_method(LSF, iterations=30)
        PS = psf.PSF_from_FFT_LSF_scipy_doublesided(LSF)
        # PS1 = psf.PSF_from_FFT_LSF(LSF)

        ax[0][2].plot(PS, ',-')
        # print PS
        ax[0][2].plot(PSo, '--')

        r = np.arange(0, PS.shape[0], 1)
        area = np.pi * r**2
        area = area[1:] - area[:-1]

        ax[1][2].plot(PS[1:] * area)
        ax[1][1].plot(np.cumsum(PS[1:] * area) / np.cumsum(PS[1:] * area)[-1])

        psf._plot_FT_norm(PS, ax[1][0], norm=True)
        psf._plot_FT_norm(PSo, ax[1][0], norm=True)
        PS = psf.PSF_2D_from_linescan(np.abs(PS))
        LSf = psf.LSF_from_PSF(PS)
        ax[0][1].plot(LSf / np.amax(LSf), '-.')

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
    # ax[0][2].semilogy()

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
