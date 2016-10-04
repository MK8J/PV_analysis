import numpy as np
import numbers

from semiconductor.material import IntrinsicCarrierDensity
from semiconductor.material import BandGapNarrowing


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


def getvalue_modelornumber(value, model, extension, **kwargs):
    '''
    test is the provded value is a float or numpy array. If it is not,
    it provides the value returned by model.extension()

    inputs:
        value:
            the value to test
        model:
            the model to use if the value is a string
        extension:
            the function of the model to be called to find the desribed value
        kwargs: (optional)
            Values to be passed to the model's extension
    '''
    if isinstance(value, numbers.Number):
        value = value
    elif isinstance(value, np.ndarray):
        value = value
    elif hasattr(model, extension) and isinstance(value, str):
        value = getattr(model, extension)(**kwargs)

    else:
        print('Incorrect type, or not a function')

    return value


class sample():

    name = None
    sample_id = None
    _dopant_type = None  # takes either 'n-type' or b'p-type'
    thickness = None
    absorptance = 1
    _Na = None
    _Nd = None
    _doping = None
    _ni = IntrinsicCarrierDensity().calculationdetails['author']
    _nieff = BandGapNarrowing().calculationdetails['author']
    temp = 300  # as most measurements are done at room temperature
    nxc = None

    def attrs(self, dic):
        '''
        sets the values in a dictionary
        '''
        assert type(dic) == dict
        for key, val in dic.items():
            if hasattr(self, key):
                setattr(self, key, val)

    @property
    def dopant_type(self):
        '''
        returns the dopant type
        '''
        return self._dopant_type

    @dopant_type.setter
    def dopant_type(self, value):
        '''
        returns the dopant type
        '''
        if value == 'p' or value == 'p-type':
            self._dopant_type = 'p-type'
        elif value == 'n' or value == 'n-type':
            self._dopant_type = 'n-type'

        self.doping = self._doping

    @property
    def doping(self):
        '''
        Returns the number of net dopants. This is not the ionised dopants
        '''
        if self._Na is None or self._Nd is None:
            doping = 0
        else:
            doping = abs(self._Na - self._Nd)
        return doping

    @doping.setter
    def doping(self, value):
        '''
        Sets the number of dopant atoms. It assumes there is only one dopant type
        i.e if it is a p-type material with 1e16 dopants, this function sets
        Na = 1e16 and Nd = 0.
        '''
        self._doping = value

        if self.dopant_type is not None:

            if self.dopant_type == 'p-type':
                self._Na = value
                self._Nd = 0
            elif self.dopant_type == 'n-type':
                self._Nd = value
                self._Na = 0
            else:
                print('\n\n', self.dopant_type, '\n\n')

    @property
    def Na(self):
        '''
        returns the number of acceptor dopant atoms
        '''
        return self._Na

    def _check_dopant_type(self):
        if self._Na > self._Nd:
            self.dopant_type = 'n-type'
        else:
            self.dopant_type = 'p-type'

    @Na.setter
    def Na(self, value):
        self._Na = value
        self._check_dopant_type()

    @property
    def Nd(self):
        '''
        returns the number of donor dopant atoms
        '''
        return self._Nd

    @Nd.setter
    def Nd(self, value):
        self._Nd = value
        self._check_dopant_type()

    @property
    def ni(self):
        model = IntrinsicCarrierDensity(
            material='Si', temp=self.temp,
        )

        return getvalue_modelornumber(self._ni, model, 'update',
                                      author=self._ni)

    @ni.setter
    def ni(self, val):
        self._ni = val

    @property
    def ni_eff(self):
        model = BandGapNarrowing(
            material='Si',
            temp=self.temp,
            nxc=self.nxc,
            Na=self.Na,
            Nd=self.Nd,
        )
        return getvalue_modelornumber(self._nieff, model, 'ni_eff', ni=self.ni,
                                      author=self._nieff)

    @ni_eff.setter
    def ni_eff(self, val):
        self._nieff = val
