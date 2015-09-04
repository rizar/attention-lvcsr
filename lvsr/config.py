import os.path
from picklable_itertools.extras import equizip
from pykwalify.core import Core
from StringIO import StringIO

import yaml

prototype = yaml.load(StringIO(
"""
data:
    batch_size: 10
    max_length:
    normalization:
    sort_k_batches:
    dataset: TIMIT
net:
    dim_dec: 100
    dims_bidir: [100]
    dims_bottom: [100]
    enc_transition: !!python/name:blocks.bricks.recurrent.SimpleRecurrent
    dec_transition: !!python/name:blocks.bricks.recurrent.SimpleRecurrent
    attention_type: content
    use_states_for_readout: False
    lm: {}
regularization:
    dropout: False
    noise:
initialization:
    -
        - /recognizer
        - weights_init
        - !!python/object/apply:blocks.initialization.IsotropicGaussian [0.1]
    -
        - /recognizer
        - biases_init
        - !!python/object/apply:blocks.initialization.Constant [0.0]
    -
        - /recognizer
        - rec_weights_init
        - !!python/object/apply:blocks.initialization.Orthogonal []
training:
    gradient_threshold: 100.0
    scale: 0.01
    momentum: 0.0
"""))


schema = """
type: map
mapping:
    data:
        mapping:
            batch_size:
                type: int
            max_length:
                type: int
                allowempty: yes
            normalization:
                type: str
                allowempty: yes
            sort_k_batches:
                type: str
                allowempty: yes
            dataset:
                type: str
    net:
        mapping:
            dim_dec:
                type: int
            dims_bidir:
                type: seq
                sequence:
                    - type: int
            dims_bottom:
                type: seq
                sequence:
                    - type: int
            enc_transition:
                type: any
            dec_transition:
                type: any
            attention_type:
                type: str
            use_states_for_readout:
                type: bool
            lm:
                type: map
                allowempty: yes
    regularization:
        mapping:
            dropout:
                type: bool
            noise:
                type: str
                allowempty: yes
    initialization:
        type: seq
        sequence:
            - type: seq
              sequence:
                  - type: any
    training:
        type: map
        mapping:
            gradient_threshold:
                type: float
            scale:
                type: float
            momentum:
                type: float
"""


def read_config(file_):
    """Reads a config from a file object.

    Merge changes from a user-made config into the prototype.
    Does not allow to create fields non-existing in the prototypes.
    Interprets merge hints such as e.g. "%extend".

    """
    config = prototype
    changes = yaml.load(file_)
    if 'parent' in changes:
        with open(os.path.expandvars(changes['parent'])) as src:
            config = read_config(src)
    merge_recursively(config, changes)
    return config


def merge_recursively(config, changes):
    for key, value in changes.items():
        pure_key = key
        hint = None
        if '%' in key:
            pure_key, hint = key.split('%')
        if isinstance(value, dict):
            if hint:
                raise ValueError
            if isinstance(config.get(pure_key), dict):
                merge_recursively(config[pure_key], value)
            else:
                config[pure_key] = value
        elif isinstance(value, list):
            if hint == 'extend':
                config[pure_key].extend(value)
            elif hint is None:
                config[pure_key] = value
            else:
                raise ValueError
        else:
            config[pure_key] = value


def make_config_changes(config, changes):
    for path, value in changes:
        parts = path.split('.')
        assign_to = config
        for part in parts[:-1]:
            assign_to = assign_to[part]
        assign_to[parts[-1]] = yaml.load(value)


def load_config(cmd_args):
    config = prototype
    if cmd_args.config_path:
        with open(cmd_args.config_path, 'rt') as src:
            config = read_config(src)
    config['cmd_args'] = cmd_args.__dict__
    make_config_changes(config, equizip(
        cmd_args.config_changes[::2],
        cmd_args.config_changes[1::2]))
    core = Core(source_data=config, schema_data=yaml.safe_load(schema))
    core.validate(raise_exception=True)
    return config
