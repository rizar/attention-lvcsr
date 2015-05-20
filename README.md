# Fully Neural LVSR

### What is available

All the code is in `lvsr`. It is structured as follows:

* the `datasets` folder contain the dataset classes. TIMIT and WSJ are available.
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

2. `cd fully-neural-lvsr && source env.sh`
   After that you can run experiments from any directory. Note: it is important  
   that Blocks and Theano are _not_ installed by `pip -e`, otherwise it is impossible
   to override them with $PYTHONPATH

3. Prepare the normalization parameters. Forcing feature means to be zero and variances 
   one has proven to be crucial to make anything work.
 
   `$LVSR/lvsr/run.py init_norm timit_delta_norm.pkl $LVSR/lvsr/configs/timit_bothgru2_fbank_qctc.yaml data.feature_name "'fbank_and_delta_delta'" data.normalization None`
 
   That will create a pickle `timit_delta_norm.pkl` in the current directory.

4. Run training. Something like this should do the job:

   ``lvsr/firsttry/main.py train timit_bothgru_cumsum.pkl lvsr/configs/timit_bothgru_cumsum.py`` 

    _norm.pkl_ should be in the same directory where training is started.

5. Use the trained model:

   ``lvsr/firsttry/main.py search timit_bothgru_cumsum_model.pkl``

   Currently it needs a GPU to beam-search, but this can fixed very quickly 
   (I just have a lot of them and don't care too much).
  

