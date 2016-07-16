
class image():
    '''
    A class to handel PL images
    '''

    image = None
    exposure = None
    JL = None
    Vtm = None
    Jtm = None
    fnames = None
    deconvolved = None

    def __init__(self, **kwargs):

        self._update(**kwargs)
        pass

    def get_attr_asdic(self):
        # dic = {
        # 'images':self.images,
        # 'exposure':self.exposure,
        # 'illumination':self.illumination,
        # 'bias':self.bias,
        # 'deconvolved':self.deconvolved
        # }
        return {
            'images': self.image,
            'exposure': self.exposure,
            'JL': self.JL,
            'Vtm': self.Vtm,
            'Jtm': self.Jtm,
            'deconvolved': self.deconvolved
        }

    def _update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)



