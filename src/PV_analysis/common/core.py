
import numpy as np


def Binning(data, BinAmount):

    if len(data.shape) != 1:
        data2 = np.zeros((data.shape[0] // BinAmount, data.shape[1]))
    else:
        data2 = np.zeros((data.shape[0] // BinAmount))

    for i in range(data.shape[0] // BinAmount):
        data2[i] = np.mean(data[i * BinAmount:(i + 1) * BinAmount], axis=0)

    return data2


def Binning_Named(data, BinAmount):

    if len(data.dtype.names) != 1:
        data2 = np.copy(data)[::BinAmount]

    for i in data.dtype.names:
        for j in range(data.shape[0] // BinAmount):
            data2[i][j] = np.mean(
                data[i][j * BinAmount:(j + 1) * BinAmount], axis=0)

    return data2


class sample():

    name = None
    sample_id = None
    dopant_type = None  # takes either 'n-type' or b'p-type'
    thickness = None
    absorptance = 1
    _Na = None
    _Nd = None
    temp = 300  # as most measurements are done at room temperature

    def attrs(self, dic):
        '''
        sets the values in a dictionary
        '''
        assert type(dic) == dict
        for key, val in dic.items():
            if hasattr(self, key):
                setattr(self, key, val)

    @property
    def doping(self):
        '''
        Returns the number of net dopants. This is not the ionised dopants
        '''

        doping = abs(self._Na - self._Nd)
        return doping

    @doping.setter
    def doping(self, value):
        '''
        Sets the number of dopant atoms. It assumes there is only one dopant type
        i.e if it is a p-type material with 1e16 dopants, this function sets
        Na = 1e16 and Nd = 0.
        '''

        if self.dopant_type is None:
            print('Doping type not set')

        else:
            if self.dopant_type == 'p-type':
                self._Na = value
                self._Nd = 0
            elif self.dopant_type == 'n-type':
                self._Nd = value
                self._Na = 0

    @property
    def Na(self):
        '''
        returns the number of acceptor dopant atoms
        '''
        return self._Na

    @Na.setter
    def Na(self, value):
        self._Na = value

    @property
    def Nd(self):
        '''
        returns the number of donor dopant atoms
        '''
        return self._Nd

    @Nd.setter
    def Nd(self, value):
        self._Nd = value
