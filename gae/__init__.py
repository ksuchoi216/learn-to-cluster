from .test_gae import test_gae
from .train_gae import train_gae

__factory__ = {
    'test_gae': test_gae,
    'train_gae': train_gae,
}


def build_handler(phase):
    key_handler = '{}_gae'.format(phase)
    if key_handler not in __factory__:
        raise KeyError("Unknown op:", key_handler)
    return __factory__[key_handler]
