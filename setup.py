from os import path
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

setup(
    name='lvsr',
    description='Fully Neural LVSR',
    url='https://github.com/bartvm/blocks',
    author='Dzmitry Bahdanau',
    license='MIT',
    packages=find_packages(exclude=['examples', 'docs', 'tests']),
    zip_safe=False
)
