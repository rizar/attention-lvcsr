import os.path
from picklable_itertools.extras import equizip
from pykwalify.core import Core

import yaml


def read_config(file_):
    """Reads a config from a file object.

    Merge changes from a user-made config into the prototype.
    Does not allow to create fields non-existing in the prototypes.
    Interprets merge hints such as e.g. "%extend".

    """
    with open('lvsr/configs/prototype.yaml') as prototype:
        config = yaml.load(prototype)
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
    with open('lvsr/configs/prototype.yaml') as prototype:
        config = yaml.load(prototype)
    if cmd_args.config_path:
        with open(cmd_args.config_path, 'rt') as src:
            config = read_config(src)
    config['cmd_args'] = cmd_args.__dict__
    make_config_changes(config, equizip(
        cmd_args.config_changes[::2],
        cmd_args.config_changes[1::2]))
    with open('lvsr/configs/config_schema.yaml') as schema:
        core = Core(source_data=config, schema_data=yaml.safe_load(schema))
    core.validate(raise_exception=True)
    return config
