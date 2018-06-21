
import numpy as np
import matplotlib.pyplot as plt


def teals_method(LSF):
    LSF /= np.amax(LSF)
    coefficient_matrix = generate_coefficient_matrix(LSF.shape[0])

    # find outermost psf value
    psf_teal = np.zeros(LSF.shape[0], dtype=float)
    psf_teal[-1] = LSF[-1] / coefficient_matrix[- 1, - 1]

    # find all psf values
    for counter1 in range(LSF.shape[0])[::-1]:
        psf_teal[counter1] = (LSF[counter1] -
                              np.sum(psf_teal[counter1:] * coefficient_matrix[counter1, (counter1):])) \
            / (coefficient_matrix[counter1, counter1])
    return psf_teal


def generate_coefficient_matrix(len_lsf, Plot=False):
    coefficient_matrix = np.zeros((len_lsf, len_lsf), dtype=float)

    for counter1 in range(len_lsf):
        x_min = (counter1 - 0.5)
        x_max = (counter1 + 0.5)

        for counter2 in range(counter1, len_lsf):
            r_min = (counter2 - 0.5)
            r_max = (counter2 + 0.5)
            coefficient_matrix[
                counter1, counter2] = 2.0 * _calculate_enclosed_area(
                    x_min, x_max, r_min, r_max)

    X = np.arange(0, coefficient_matrix.shape[0], dtype=np.float)
    Y = np.arange(0, coefficient_matrix.shape[1], dtype=np.float)
    X, Y = np.meshgrid(X, Y)
    if Plot:
        fig2 = plt.figure(2)
        ax = fig2.add_subplot(111, projection='3d')
        ax.plot_surface(
            X, Y, np.log(coefficient_matrix), rstride=10, cstride=10)

        plt.show()
    return coefficient_matrix


def _teal_ana_solution(r, x):
    if r < x:
        ret = np.nan
    else:
        ret = 0.5 * (x * np.sqrt(r**2 - x**2) + r**2 *
                     np.arctan(x / np.sqrt(r**2 - x**2)))
    return ret


def _calculate_enclosed_area(x_min, x_max, r_min, r_max):
    '''
    this function returns the area enclosed by the input bounding area
    it used the formula:
    integral (sqrt(r_max^2-x^2) dx - integral (sqrt(r_min^2-x^2) dx
    the limits of the integral are x_max,and x_min
    an analytical solution exists for the integral(sqrt(r^2-x^2) dx
    = 1/2(x.sqrt(r^2-x^2)+r^2*np.arctan(x/sqrt(r^2-x^2)))
    '''

    abc = _teal_ana_solution(
        r_max, x_max) - _teal_ana_solution(r_max, x_min)
    asc = _teal_ana_solution(
        r_min, x_max) - _teal_ana_solution(r_min, x_min)

    if np.isnan(asc):
        asc = 0

    if np.isnan(abc):
        print(abc)

    enclosed_area = abc - asc

    if abc < 0:
        print(enclosed_area, abc, asc)

    return enclosed_area
