
import numpy as np
import json


def save_named_data(fname, data, settings):

    _save_data_named(fname + '.pv_dat', data)
    _save_settings(fname + '.inf', settings)


def _save_data_named(fname, data):

    np.savetxt(fname, data, header=' '.join(data.dtype.names), delimiter='\t')


def _save_settings(fname, settings):
    '''
    writes data to string
    '''
    assert isinstance(settings, dict)

    with open(fname, 'w') as f:
        f.write(json.dumps(settings, sort_keys=True, indent=4))
