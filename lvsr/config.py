import copy
import os.path
from collections import OrderedDict

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


class Configuration(dict):
    """Convenient access to a multi-stage configuration.

    Attributes
    ----------
    multi_stage : bool
        ``True`` if the configuration describes multiple training stages
    ordered_stages : OrderedDict, optional
        Configurations for all the training stages in the order of
        their numbers.

    """
    def __init__(self, config_path, schema_path, config_changes):
        with open(config_path, 'rt') as src:
            config = read_config(src)
        make_config_changes(config, config_changes)

        self.multi_stage = 'stages' in config
        if self.multi_stage:
            stages = [(k, v) for k, v in config['stages'].items() if v]
            ordered_changes = OrderedDict(
                sorted(stages, key=lambda (k, v): v['number'],))
            self.ordered_stages = OrderedDict()
            for name, changes in ordered_changes.items():
                current_config = copy.deepcopy(config)
                del current_config['stages']
                del changes['number']
                merge_recursively(current_config, changes)
                self.ordered_stages[name] = current_config

        # Validate the configuration and the training stages
        with open(os.path.expandvars(schema_path)) as schema_file:
            schema = yaml.safe_load(schema_file)
            core = Core(source_data=config, schema_data=schema)
            core.validate(raise_exception=True)
            if self.multi_stage:
                for stage in self.ordered_stages.values():
                    core = Core(source_data=stage, schema_data=schema)
                    core.validate(raise_exception=True)
        super(Configuration, self).__init__(config)
