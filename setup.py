"""Installation script."""
from os import path
from setuptools import find_packages, setup

HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, 'README.rst')) as f:
    LONG_DESCRIPTION = f.read().strip()

setup(
    name='picklable-itertools',
    version='0.1.1a0',  # PEP 440 compliant
    description='itertools. But picklable. Even on Python 2.',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/mila-udem/picklable-itertools',
    author='David Warde-Farley',
    author_email='d.warde.farley@gmail.com',
    license='MIT',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Utilities',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
    ],
    keywords='pickle serialize pickling itertools iterable iteration',
    packages=find_packages(exclude=['tests']),
    install_requires=['six'],
    zip_safe=True
)
