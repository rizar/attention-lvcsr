Blocks extras
=============

The ``blocks-extras`` repository contains a variety of useful extensions to
Blocks_. The goal of this repository is for people to share useful extensions,
bricks, and other code snippets that are perhaps not general enough to go into
the core Blocks repository.

.. _Blocks: https://github.com/bartvm/blocks

Installation
------------

Clone to a directory of your choice.

.. code-block:: bash

   $ git clone git@github.com:mila-udem/blocks-extras.git

Because of `limitations in pip`_ it is important that you install ``blocks-extras``
the same way that you installed Blocks. So, if you installed Blocks in editable mode,
use the command:

.. code-block:: bash

   $ pip install -e .
   
And if you installed Blocks in the normal mode (so using ``pip install`` without ``-e``)
then run this from the directory you just cloned instead:

.. code-block:: bash

   $ pip install .
   
Note that you `might have problems`_ with namespace packages if you try to install using
``python setup.py develop``.

.. _limitations in pip: https://github.com/pypa/pip/issues/3
.. _might have problems: https://github.com/pypa/packaging-problems/issues/12

Usage
-----

.. code-block:: python

   from blocks_extras.extensions.plotting import Plot
