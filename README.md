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


1. Make sure that `$FUEL_DATA_PATH/timit` contains `timit.h5` and `phonemes.pkl` (which can
   be found at `/data/lisa/data/timit/readable/phonemes.pkl`)

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

4. Run training:

   `$LVSR/lvsr/run.py train timit_bothgru2_fbank_lm_mn.pkl $LVSR/lvsr/configs/timit_bothgru2_fbank_qctc.yaml net.use_states_for_readout True net.birnn_depth 2 regularization.max_norm 1 regularization.noise 0.05`

5. Use the trained model:

   `lvsr/firsttry/main.py search --part=test timit_bothgru2_fbank_qctc_noise_model.pkl  $LVSR/lvsr/configs/timit_bothgru2_fbank_qctc.yaml` 

   A GPU is needed unless `https://github.com/Theano/Theano/pull/2339` is used.

Don't hesitate to contact for more information!
