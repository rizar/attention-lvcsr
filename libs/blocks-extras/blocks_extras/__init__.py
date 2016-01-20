from blocks.config import config

config.add_config('bokeh_server', type_=str, default='http://localhost:5006/')
config.load_yaml()
