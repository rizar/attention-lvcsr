# Fully Neural LVSR

Dependencies:

- Theano (please install from the submodule `theano`)
- blocks (please install from the submodule `blocks`)
- Fuel
- blocks-extras

As of June 20 the submodule versions are not so much different from the
respective master branches. `theano` contains a very useful PR by Frederic
Bastien that allows to unpickled GPU-based shared variables on CPU. `blocks`
has a PR merged which is currently waiting for review. Unfortunately 
simply adding the `blocks` folder to ``PYTHONPATH`` will not cut it 
because of the tricks required to make Blocks a namespace package:
you have to install it (at least in a virtual environment).

### What is available

All the code is in `lvsr`. It is structured as follows:

* `main.py` contains most of the code, `run.py` is the script to run

* `attention.py` contains the attention mechanism with convolutional features

* the `configs` folder contain experiment configurations

* `error_rate.py` : Levenshtein distance and WER

* `expressions.py` : nice pieces of Theano code such as monotonicity penalty, 
   weights entropy, 1-D convolution

* `bricks.py` : additional bricks, in fact only one so far

* `config.py` : hierarhical configuration support

##### Almost obsolete

This code will become unnecessary as soon as preprocessed ``H5PyDataset`` compatible
tables are available.

* `preprocessing.py` contains implemented preprocessings, the only is available so far is
  `log_spectrogram`. 

* the `datasets` folder contains the dataset classes. TIMIT and WSJ are available.
  It expects hdf5 tables in `$FUEL_DATA_PATH`, the one for TIMIT is called
  `timit.hdf5` and can be found at `/data/lisatmp3/bahdanau/timit.h5`.

### Usage

First set important envinronment variables:

`git clone https://github.com/rizar/fully-neural-lvsr.git`

`source env.sh`

#### WSJ

1. Make sure that `$FUEL_DATA_PATH/wsj` contains `wsj.h5`

2. Prepare the normalization parameters: 

   `$LVSR/lvsr/run.py init_norm wsj_norm.pkl _ data.datasets "'WSJ'"`

3. Run training:

   `$LVSR/lvsr/run.py train wsj_good.zip $LVSR/lvsr/configs/wsj_good.yaml`

4. Test:

   `$LVSR/lvsr/run.py search --part=test wsj_good.zip  $LVSR/lvsr/configs/wsj_good.yaml`

Don't hesitate to contact for more information!

#### TIMIT

##### Slightly outdated!

1. Make sure that `$FUEL_DATA_PATH/timit` contains `timit.h5` and `phonemes.pkl` (which can
   be found at `/data/lisa/data/timit/readable/phonemes.pkl`)

2. Prepare the normalization parameters. Forcing feature means to be zero and variances 
   one has proven to be crucial to make anything work.

   `$LVSR/lvsr/run.py init_norm timit_delta_norm.pkl $LVSR/lvsr/configs/timit_bothgru2_fbank_qctc.yaml data.feature_name "'fbank_and_delta_delta'" data.normalization None`
 
   That will create a pickle `timit_delta_norm.pkl` in the current directory.

3. Run training:

   `$LVSR/lvsr/run.py train timit_bothgru2_fbank_lm_mn.pkl $LVSR/lvsr/configs/timit_bothgru2_fbank_qctc.yaml net.use_states_for_readout True net.birnn_depth 2 regularization.max_norm 1 regularization.noise 0.05`

4. Use the trained model:

   `lvsr/firsttry/main.py search --part=test timit_bothgru2_fbank_qctc_noise_model.pkl  $LVSR/lvsr/configs/timit_bothgru2_fbank_qctc.yaml` 

   A GPU is needed unless `https://github.com/Theano/Theano/pull/2339` is used.
