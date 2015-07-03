.. image:: https://img.shields.io/coveralls/mila-udem/picklable-itertools.svg
   :target: https://coveralls.io/r/mila-udem/picklable-itertools

.. image:: https://travis-ci.org/mila-udem/picklable-itertools.svg?branch=master
   :target: https://travis-ci.org/mila-udem/picklable-itertools

.. image:: https://img.shields.io/scrutinizer/g/mila-udem/picklable-itertools.svg
   :target: https://scrutinizer-ci.com/g/mila-udem/picklable-itertools/

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/mila-udem/picklable-itertools/blob/master/LICENSE

picklable-itertools
===================

A reimplementation of the Python standard library's ``itertools``, in Python,
using picklable iterator objects. Intended to be Python 2.7 and 3.4+
compatible. Also includes picklable, Python {2, 3}-compatible implementations
of some related utilities, including some functions from the Toolz_ library,
in ``picklable_itertools.extras``.

.. _Toolz: http://toolz.readthedocs.org/en/latest/

Why?
----
* Because the standard library pickle module (nor the excellent dill_ package)
  can't serialize all of the ``itertools`` iterators, at least on Python 2
  (at least some appear to be serializable on Python 3).
* Because there are lots of instances where these things in ``itertools`` would
  simplify code, but can't be used because serializability must be maintained
  across both Python 2 and Python 3.  The in-development framework Blocks_ is
  our first consumer. We'd like to be able to serialize the entire state of a
  long-running program for later resumption. We can't do this with
  non-picklable objects.

.. _dill: https://github.com/uqfoundation/dill
.. _blocks: https://github.com/bartvm/blocks

Philosophy
----------
* *This should be a drop-in replacement.* Pretty self-explanatory. Test
  against the standard library ``itertools`` or builtin implementation to
  verify behaviour matches. Where Python 2 and Python 3 differ in their
  naming, (``filterfalse`` vs ``ifilterfalse``, ``zip_longest`` vs. ``izip_longest``)
  we provide both. We also provide names that were only available in the
  Python 2 incarnation of ``itertools`` (``ifilter``, ``izip``), also available
  under their built-in names in Python 3 (``filter``, ``zip``), for convenience.
  As new objects are added to the Python 3 ``itertools`` module, we intend
  to add them (``accumulate``, for example, appears only in Python 3, and a
  picklable implementation is contained in this package.)
* *Handle built-in types gracefully if possible.* List iterators, etc.
  are not picklable on Python 2.x, so we provide an alternative
  implementation. File iterators are handled transparently as well. dict
  iterators and set iterators are currently *not* supported.
  ``picklable_itertools.xrange`` can be used as a drop-in replacement for
  Python 2 ``xrange``/Python 3 ``range``, with the benefit that the iterators
  produced by it will be picklable on both Python 2 and 3.
* *Premature optimization is the root of all evil.* These things are
  implemented in Python, so speed is obviously not our primary concern. Several
  of the more advanced iterators are constructed by chaining simpler iterators
  together, which is not the most efficient thing to do but simplifies the
  code a lot. If it turns out that speed (or a shallower object graph) is
  necessary or desirable, these can always be reimplemented. Pull requests
  to this effect are welcome.
