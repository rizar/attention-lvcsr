# Attention-based Speech Recognizer

The reference implementation for the papers

End-to-End Attention-based Large Vocabulary Speech Recognition.
_Dzmitry Bahdanau, Jan Chorowski, Dmitriy Serdyuk, Philemon Brakel, 
Yoshua Bengio_
([arxiv draft](http://arxiv.org/pdf/1508.04395), ICASSP 2016)
and
Task Loss Estimation for Sequence Prediction.
_Dzmitry Bahdanau, Dmitriy Serdyuk, Philémon Brakel, Nan Rosemary Ke, 
Jan Chorowski, Aaron Courville, Yoshua Bengio_
([arxiv draft](http://arxiv.org/pdf/1511.06456), submitted to ICLR 2016).


### How to use

- install all the dependencies (see the list below)
- set your environment variables by calling `source env.sh`

Then, please proceed to [`exp/wsj`](exp/wsj/README.md) for the instructins how
to replicate our results on Wall Street Journal (WSJ) dataset
(available  at  the  Linguistic  Data  Consortium as LDC93S6B and LDC94S13B).

### Dependencies

- Python packages: pykwalify, toposort, pyyaml, numpy, pandas, pyfst;
- [kaldi](https://github.com/kaldi-asr/kaldi);
- [kaldi-python](https://github.com/dmitriy-serdyuk/kaldi-python).

Given that you have the dataset in HDF5 format, the models can be trained
without Kaldi and PyFst.

### Installation

- Compile Kaldi. 
  It should be compiled with 
  `--shared` option, it means that Kaldi should be configured like
  ```
  ./configure --shared
  ```
  we need Kaldi to be compiled in shared mode to be able to use kaldi-python.

  We don't train anything with Kaldi, so there is no need to compile it
  with cuda, so if you have any problems with Kaldi+CUDA, feel free to
  turn it off:
  ```
  ./configure --shared --use-cuda=no
  ```
  After this step you should have openfst installed at `$KALDI_ROOT/tools/openfst`.
- Install python packages.
  You can use pip for that:
  ```
  pip install pykwalify toposort pyyaml numpy pandas pyfst
  ```
- Install kaldi-python.
  Clone the repository and run
  ```
  python setup.py install
  ```
  kaldi-python will be compiled and installed to your system, you can check that 
  everything went right by running
  ```
  python -c "import kaldi_io"
  ```

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
