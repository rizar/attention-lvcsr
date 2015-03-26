from abc import ABCMeta, abstractmethod

# My custom configuration language - sorry for this crap, but
# 'state' for solution from groundhog does not have features
# I need and I was too lazy to find a proper solution.

class BaseConfig(object):
    # Python 3 goes to hell!
    __metaclass__ = ABCMeta

    @abstractmethod
    def merge(self, other):
        pass


class Config(BaseConfig, dict):

    def merge(self, other):
        for key in other:
            if isinstance(self.get(key), BaseConfig):
                self[key].merge(other[key])
            else:
                self[key] = other[key]


class InitList(BaseConfig, list):

    def merge(self, other):
        self.extend(other)


def default_config():
    return Config(
        net=Config(
            dim_dec=100, dim_bidir=100, dims_bottom=[100],
            enc_transition='SimpleRecurrent',
            dec_transition='SimpleRecurrent',
            attention_type='content',
            use_states_for_readout=False),
        regularization=Config(
            dropout=False,
            noise=None),
        initialization=InitList([
            ('/recognizer', 'weights_init', 'IsotropicGaussian(0.1)'),
            ('/recognizer', 'biases_init', 'Constant(0.0)'),
            ('/recognizer', 'rec_weights_init', 'Orthogonal()')]),
        data=Config(
            batch_size=10,
            normalization=None
        ))
