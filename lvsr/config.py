import os.path
from pykwalify.core import Core

import yaml


def read_config(file_):
    """Reads a config from a file object.

    Merge changes from a user-made config into the prototype.
    Does not allow to create fields non-existing in the prototypes.

    """
    config = yaml.load(file_)
    if 'parent' in config:
        with open(os.path.expandvars(config['parent'])) as src:
            changes = dict(config)
            config = read_config(src)
            merge_recursively(config, changes)
    return config


def merge_recursively(config, changes):
    for key, value in changes.items():
        if isinstance(value, dict):
            if isinstance(config.get(key), dict):
                merge_recursively(config[key], value)
            else:
                config[key] = value
        else:
            config[key] = value


def make_config_changes(config, changes):
    for path, value in changes:
        parts = path.split('.')
        assign_to = config
        for part in parts[:-1]:
            assign_to = assign_to[part]
        assign_to[parts[-1]] = yaml.load(value)


def load_config(config_path, schema_path, config_changes):
    with open(config_path, 'rt') as src:
        config = read_config(src)
    make_config_changes(config, config_changes)
    with open(os.path.expandvars(schema_path)) as schema:
        core = Core(source_data=config, schema_data=yaml.safe_load(schema))
    core.validate(raise_exception=True)
    return config
