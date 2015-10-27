# Attention-based Speech Recognizer

The reference implementation for the paper

End-to-End Attention-based Large Vocabulary Speech Recognition.
_Dzmitry Bahdanau, Jan Chorowski, Dmitriy Serdyuk, Philemon Brakel, Yoshua Bengio_.

([arxiv draft](http://arxiv.org/pdf/1508.04395), submitted to ICASSP 2016).

### How to use

- install all the dependencies (see the list below)
- set your environment variables by calling `source env.sh`

Then, please proceed to [`exp/wsj`](exp/wsj/README.md) for the instructins how
to replicate our results on Wall Street Journal (WSJ) dataset
(available  at  the  Linguistic  Data  Consortium as LDC93S6B and LDC94S13B).

### Dependencies

- Python packages: pykwalify, toposort, pyyaml, numpy, pandas, pyfst
- [kaldi](https://github.com/kaldi-asr/kaldi)
- [kaldi-python](https://github.com/dmitriy-serdyuk/kaldi-python)

Given that you have the dataset in HDF5 format, the models can be trained
without Kaldi and PyFst

### Subtrees

The repository contains custom modified versions of Theano, Blocks, Fuel,
picklable-itertools, Blocks-extras as [subtrees]
(http://blogs.atlassian.com/2013/05/alternatives-to-git-submodule-git-subtree/).
In order to ensure that these
specific versions are used, we recommend to **uninstall regular installations
of these packages if you have them installed** in addition to sourcing
`env.sh`.

### License

MIT
