import sys
import numpy as np
import openpyxl as pyxl

from ..core import lifetime_Voc as LTC


def load_data(file_path, voc_header='Voc', gen_header='Gen', time_header='time_s', raw_dat_ext='.Raw Data.dat', inf_ext='.inf'):
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
    inf = extract_info(file_path.replace(raw_dat_ext, inf_ext))

    # pass to the lifetime class
    try:
        ltc.time = data[time_header]
        ltc.V = data[voc_header]
        ltc.gen_V = data[gen_header]
    except:
        print('Could not find passed headers\n'
              'Please check you entered the correct headers for the file'
              'The headers in the file were {0}'.format(data.dtype.names))

    if 'reflection' in inf.keys():
        inf['absorptance'] = (100. - inf['reflection']) / 100.

    if 'type' in inf.keys():
        inf['dopant_type'] = inf['type']

    if 'fs' in inf.keys():
        inf['Fs'] = inf['fs']

    # Pass a dic to update atttrs, but turn off warnings
    # for non attributes first
    ltc._warnings = False
    ltc.attrs = inf
    # turns the warnings back on
    ltc._warnings = True

    return ltc


def extract_measurement_data(file_path):
    data = np.genfromtxt(
        file_path, unpack=True, names=True, delimiter='\t')

    return data


def extract_info(file_path):
    '''
    returns a dic with all names in lower case

    assumes the file is in the format:
    item_name:\tvalue or string\n
    '''
    List = {}

    with open(file_path, 'r') as f:
        s = f.read()

    if '\n\n' in s:
        s = s.replace('\n\n', '\n')

    for i in s.split('\n')[2:-1]:
        # print(i)
        try:
            List[i.split(':\t')[0].strip().lower()] = num(i.split(':\t')[1])
        except:
            pass

    return List


def num(s):
    '''
    converts s to a number, or returns s
    '''
    try:
        return float(s)
    except ValueError:
        return s
