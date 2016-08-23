
import sys
import numpy as np
import openpyxl as pyxl

from ..core import lifetime_Voc as LTC


def load_data(file_path):
    '''
    Loads a tab spaced text file and
    returns a lifetime class. It assumes the lifetime a tab delimited file,
    with the headers 'Time_s', 'Voc', 'Gen'. The load will also look for a
    file with measurement information, by changing the measurement data extension to .inf. The inf file is assumed to have for format:
        property1: value
        property2: value
    '''

    # define the lifetime class
    ltc = LTC()

    # get the measurement data
    data = extract_measurement_data(file_path)
    inf = extract_info(file_path.replace('.dat', '.inf'))

    # pass to the lifetime class
    ltc.time = data['Time_s']
    ltc.V = data['Voc']
    ltc.gen_V = data['Gen']

    # Pasa a dic to update atttrs, but turn off warnings
    # for non attributes first
    ltc._warnings = False
    ltc.attrs = inf
    # turns the warnings back on
    ltc._warnings = True

    return ltc


def extract_measurement_data(file_path):
    data = np.genfromtxt(
        file_path, unpack=True, names=True, delimiter='\t')
    # s = np.array([])
    # dic = {'Time_s': 'Time', 'Generation_V': 'Gen',
    #        'PL_V': 'PL', 'PC_V': 'PC'}
    # # print np.array(data.dtype.names)
    # for i in np.array(data.dtype.names):
    #     # print i,dic[i]
    #     s = np.append(s, dic[i])
    #
    # # print s
    #
    # ('Time','Gen','PL','PC')
    return data


def extract_info(file_path):

    List = {}

    with open(file_path, 'r') as f:
        s = f.read()

    for i in s.split('\n')[2:-1]:
        # print(i)
        List[i.split(':\t')[0].strip()] = num(i.split(':\t')[1])

    return List


def num(s):
    '''
    converts s to a number, or returns s
    '''
    try:
        return float(s)
    except ValueError:
        return s
