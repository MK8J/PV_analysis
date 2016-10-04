
import numpy as np


def timeseries(data, value, correction='points', side='front'):
    '''
    background corrects a time series of data. The data is assumed a number array.

    if correction is None, no correction is performed.
    '''
    func_dic = {
        'points': _bgc_nopints,
        'percent': _bgc_percent
    }

    # it the correction is turned on
    if correction:
        if side == 'front':
            data = data[::-1]

        data = func_dic[correction](data, value)

        if side == 'front':
            data = data[::-1]

    return data


def _bgc_percent(data, percent):
    '''
    subtracts the average of the last percentae of data
    '''
    assert percent < 100
    data = np.copy(data)
    index = int(percent * data.shape[0] / 100)
    data -= np.average(data[-index:])

    return data


def _bgc_nopints(data, no_points):
    '''
    background subtracts from the provided number of data points
    '''
    assert data.shape[0] > no_points
    data = np.copy(data)
    data -= np.average(data[-no_points:])

    return data
