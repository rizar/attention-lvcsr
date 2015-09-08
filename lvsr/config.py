import os.path
from pykwalify.core import Core

import yaml


def read_config(file_):
    """Reads a configuration from YAML file.

    Resolves parent links in the configuration.

    """
    config = yaml.load(file_)
    if 'parent' in config:
        with open(os.path.expandvars(config['parent'])) as src:
            changes = dict(config)
            config = read_config(src)
            merge_recursively(config, changes)
    return config


def merge_recursively(config, changes):
    """Merge hierarchy of changes into a configuration."""
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            merge_recursively(config[key], value)
        else:
            config[key] = value


def make_config_changes(config, changes):
    """Apply changes to a configuration.

    Parameters
    ----------
    config : dict
        The configuration.
    changes : dict
        A dict of (hierachical path as string, new value) pairs.

    """
    for path, value in changes:
        parts = path.split('.')
        assign_to = config
        for part in parts[:-1]:
            assign_to = assign_to[part]
        assign_to[parts[-1]] = yaml.load(value)


def safe_compile_configuration(config_path, schema_path, config_changes):
    """Read configuration, apply changes, validate."""
    with open(config_path, 'rt') as src:
        config = read_config(src)
    make_config_changes(config, config_changes)
    with open(os.path.expandvars(schema_path)) as schema:
        core = Core(source_data=config, schema_data=yaml.safe_load(schema))
    core.validate(raise_exception=True)
    return config
