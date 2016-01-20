from setuptools import find_packages, setup

setup(
    name='blocks_extras',
    install_requires=['blocks'],
    packages=find_packages(),
    scripts=['bin/blocks-plot'],
    extras_require={'plot': ['bokeh']},
    zip_safe=False
)
