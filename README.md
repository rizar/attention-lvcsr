# Fully Neural LVSR

### What is available

All the code is in `lvsr`. It is structured as follows:

* the `datasets` folder contains the dataset classes. TIMIT and WSJ are available.
  It expects hdf5 tables in `$FUEL_DATA_PATH`, the one for TIMIT is called
  `timit.hdf5` and can be found at `/data/lisatmp3/bahdanau/timit.h5`

* the `configs` folder contain experiment configurations

* `main.py` contains most of the code, `run.py` is the script to run

* `attention.py` contains different attention mechanisms tried. Warning: low code quality, 
  lots of copy-pasted code. 

* `preprocessing.py` contains implemented preprocessings, the only is available so far is
  `log_spectrogram`

* `error_rate.py` : Levenshtein distance and WER

* `expressions.py` : nice pieces of Theano code such as monotonicity penalty, weights entropy

### How to use it


1. Make sure that `$FUEL_DATA_PATH/timit` contains `timit.h5`


2. `git clone https://github.com/rizar/fully-neural-lvsr.git`

   `cd fully-neural-lvsr`

   `git submodule init`

   `git submodule update`

   `source env.sh`

   After that you can run experiments from any directory. _Note:_ it is important  
   that Blocks and Theano are _not_ installed by `pip -e`, otherwise it is impossible
   to override them with `$PYTHONPATH`

3. Prepare the normalization parameters. Forcing feature means to be zero and variances 
   one has proven to be crucial to make anything work.

   `$LVSR/lvsr/run.py init_norm timit_delta_norm.pkl $LVSR/lvsr/configs/timit_bothgru2_fbank_qctc.yaml data.feature_name "'fbank_and_delta_delta'" data.normalization None`
 
   That will create a pickle `timit_delta_norm.pkl` in the current directory.

4. Run training with max column constraint:

   `$LVSR/lvsr/run.py train timit_bothgru2_fbank_qctc_maxnorm.pkl $LVSR/lvsr/configs/timit_bothgru2_fbank_qctc.yaml regularization.max_norm 1`

   The training progress can be tracked with Bokeh (don't forget to have `bokeh-server` running!).
   When log-likelihood stops improving, restart with weight noise:

   `$LVSR/blocks/bin/blocks-dump timit_bothgru2_fbank_qctc_maxnorm.pkl`

   `$LVSR/lvsr/run.py train --params timit_bothgru2_fbank_qctc_maxnorm/params.npz timit_bothgru2_fbank_qctc_noise.pkl $LVSR/lvsr/configs/timit_bothgru2_fbank_qctc.yaml regularization.noise 0.05 training.scale 0.005`

    I am in progress of figuring out if both regularizations can be used throughout the training process.

5. Use the trained model:

   `lvsr/firsttry/main.py search --part=test timit_bothgru2_fbank_qctc_noise_model.pkl  $LVSR/lvsr/configs/timit_bothgru2_fbank_qctc.yaml` 

   A GPU is needed unless `https://github.com/Theano/Theano/pull/2339` is used.

Don't hesitate to contact for more information!
