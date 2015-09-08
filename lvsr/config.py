import os.path
from pykwalify.core import Core

import yaml

PROTOTYPE_FILE = '$LVSR/lvsr/configs/prototype.yaml'
SCHEMA_FILE = '$LVSR/lvsr/configs/schema.yaml'


def read_config(file_):
    """Reads a config from a file object.

    Merge changes from a user-made config into the prototype.
    Does not allow to create fields non-existing in the prototypes.

    """
    with open(os.path.expandvars(PROTOTYPE_FILE)) as prototype:
        config = yaml.load(prototype)
    changes = yaml.load(file_)
    if 'parent' in changes:
        with open(os.path.expandvars(changes['parent'])) as src:
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


def load_config(config_path, cmd_dict, config_changes):
    with open(os.path.expandvars(PROTOTYPE_FILE)) as prototype:
        config = yaml.load(prototype)
    if config_path:
        with open(config_path, 'rt') as src:
            config = read_config(src)
    config['cmd_args'] = cmd_dict
    make_config_changes(config, config_changes)
    with open(os.path.expandvars(SCHEMA_FILE)) as schema:
        core = Core(source_data=config, schema_data=yaml.safe_load(schema))
    core.validate(raise_exception=True)
    return config
