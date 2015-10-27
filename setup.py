from os import path
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

setup(
    name='lvsr',
    description='Fully Neural LVSR',
    url='https://github.com/rizar/fully-neural-lvsr',
    author='Dzmitry Bahdanau',
    license='MIT',
    packages=find_packages(exclude=['examples', 'docs', 'tests']),
    zip_safe=False,
    install_requires=['numpy', 'pykwalify', 'toposort', 'pyyaml',
                      'picklable-itertools', 'pandas', 'pyfst']
)
