
import sys
import os
import numpy as np
from PV_analysis.lifetime.QSSPL.core import lifetime_PL
import re


def load(file_path):

    file_ext_dic = {
        '.Raw Data.dat': 'python',
        '_Raw Data.dat': 'labview',
        '.tsv': 'tempdep'
    }

    i = 0

    tau_PL = lifetime_PL()

    for key in file_ext_dic.keys():
        if key in file_path:
            method = file_ext_dic[key]
            file_inf = file_path.replace(key, '.inf')
        else:
            i += 1

    # test if the file is a known type
    if i == len(file_ext_dic):
        print('File not supported {0}'.format(file_path))
        raise
    else:
        data = extract_raw_data(file_path, method)

        tau_PL.I_PL = data['PL']
        tau_PL.time = data['Time']
        tau_PL.gen_V = data['Gen']
        # try:
        tau_PL._m_settings = extract_info(file_inf, method)

        try:
            tau_PL.sample.temp = tau_PL._m_settings['Temp']

            tau_PL.sample.dopant_type = tau_PL._m_settings['Type'] + '-type'
            tau_PL.sample.absorptance = 1. - \
                tau_PL._m_settings['Reflection'] / 100.
            tau_PL.Fs = tau_PL._m_settings['Fs']
            tau_PL.Ai = tau_PL._m_settings['Ai']
            tau_PL.sample.thickness = tau_PL._m_settings['Thickness']
            tau_PL.sample.doping = tau_PL._m_settings['Doping']
        except:
            print('Error in loading inf PL\'s inf file')
    return tau_PL


def extract_raw_data(file_path, method):
    '''
    This grabs the raw data and other data if it can be found
    This assume the inf file has the same name as the data file.
    '''

    # Externsion of supported files
    file_ext_dic = {
        'python': _Load_RawData_Python,
        'labview': _Load_RawData_LabView,
        'tempdep': _Load_RawData_TempDep
    }

    data = None

    try:
        data = file_ext_dic[method](file_path)

    except:
        print('Error when reading raw data file: {0}'.format(file_path))
        raise

    return data


def extract_processed_data(file_path):
    '''
    This grabs the calculated data from sinton
    Extracts columns A-G of the raw data page.
    Can't extract columns H-I as it is not complete data and this would take
    some more work
    This outputs the data as a structured array, with the names as the column
    '''

    data = np.genfromtxt(file_path, delimiter='\t', names=True)

    return data


def extract_info(file_path, method):
    '''
    This grabs the measurement infromation from an inf file.
    '''

    # Externsion of supported files
    file_ext_dic = {
        'python': _Load_InfData_Python,
        'labview': _Load_InfData_LabView,
        'tempdep': _Load_InfData_TempDep
    }

    settings = {}
    settings = file_ext_dic[method](file_path)

    try:
        settings = file_ext_dic[method](file_path)
    except:
        print('Error when reading  inf file {0}'.format(file_path))
        print('No settings loaded')

    return settings


def _Load_RawData_LabView(self):
    data = np.genfromtxt(os.path.join(Directory, RawDataFile),
                         names=('Time', 'PC', 'Gen', 'PL'))

    return data


def _Load_InfData_LabView(file_path):
    '''info from inf file '''

    Cycles, dump, Frequency, LED_Voltage, dump, dump, dump, dump, DataPoints, dump = np.genfromtxt(
        Directory + InfFile, skip_header=20, skip_footer=22, delimiter=':', usecols=(1), autostrip=True, unpack=True)
    Waveform, LED_intensity = np.genfromtxt(
        Directory + InfFile, skip_header=31, skip_footer=20, delimiter=':', usecols=(1), dtype=None, autostrip=True, unpack=True)

    l = np.genfromtxt(
        Directory + InfFile, skip_header=36, delimiter=':', usecols=(1))

    Doping = l[9]
    Ai = l[6]
    Fs = l[7]
    Thickness = l[12]
    Quad = l[12]
    Lin = l[12]
    Const = 0

    Binning = int(l[2])
    Reflection = (1 - l[16]) * 100

    dic = locals()

    del dic['self']
    del dic['l']
    del dic['dump']

    return dic


def _Load_QSSPL_File_LabView_ProcessedData_File(file_path):
    return np.genfromtxt(file_path, usecols=(0, 1, 8, 9),
                         unpack=True, delimiter='\t',
                         names=('Deltan_PC', 'Tau_PC', 'Deltan_PL', 'Tau_PL'))


def _Load_RawData_Python(file_path):
    data = np.genfromtxt(
        file_path, unpack=True, names=True, delimiter='\t')
    s = np.array([])
    dic = {'Time_s': 'Time', 'Generation_V': 'Gen',
           'PL_V': 'PL', 'PC_V': 'PC'}
    # print np.array(data.dtype.names)
    for i in np.array(data.dtype.names):
        # print i,dic[i]
        s = np.append(s, dic[i])

    # print s

    data.dtype.names = s
    # ('Time','Gen','PL','PC')
    return data


def _Load_InfData_Python(file_path):

    List = {}

    with open(file_path, 'r') as f:
        s = f.read()

    s = re.sub('\n\n+', '\n', s)
    for i in s.split('\n')[2:-1]:
        if len(i) > 1:
            List[i.split(':\t')[0].strip()] = num(i.split(':\t')[1])

    return List


def Load_Python_ProcessedData_File(self):
    print('Still under construction')

    return zeros(4, 4)


def _Load_RawData_TempDep(self):
    '''
    Loads the measured data from the data file.
    This has the file extension tsv (tab seperated values)

    from a provided file name,
    takes data and outputs data with specific column headers
    '''

    # get data, something stange was happening with os.path.join
    file_location = os.path.normpath(
        os.path.join(Directory, RawDataFile))

    data = np.genfromtxt(
        os.path.join(file_location),
        unpack=True, names=True, delimiter='\t')

    # string to convert file names to program names
    dic = {'Time_s': 'Time', 'Generation_V': 'Gen',
           'PL_V': 'PL', 'PC_V': 'PC'}

    # create empty array
    s = np.array([])

    # build array of names, in correct order
    for i in np.array(data.dtype.names):
        s = np.append(s, dic[i])

    # assign names
    data.dtype.names = s

    return data


def num(s):
    '''
    converts s to a number, or returns s
    '''
    try:
        return float(s)
    except ValueError:
        return s


def _Load_InfData_TempDep(file_path):

    temp_list = {}

    with open(file_path, 'r') as f:
        file_contents = f.read()
        List = json.loads(file_contents)

    List.update(temp_list)

    return List
