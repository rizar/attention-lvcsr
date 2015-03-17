# Fully Neural LVSR

### What is available

All the code is in _lvsr_. It is structured as follows:

* the _datasets_ folder contain the dataset classes. Currently only TIMIT is available.
  It expects data in the format from _/data/lisa/data/timit/readable_, that is already
  nicely packed into numpy arrays.

* the _configs_ folder contain experiment configuration in the format understood by 
  the _firsttry_

* _firsttry_ contains the do-it-all script splitted into _main.py_ and \_\_init.py\_\_ , as
  it has be to done to ensure that unpickling of the main loop object goes smoothly.

* _attenion.py_ contains different attention mechanisms tried. Warning: low code quality, 
  lots of copy-pasted code. 

* _preprocessing.py_ contains implemented preprocessings, the only is available so far is
  log\_spectrogram

* _error\_rate.py_ : Levenshtein distance and WER

* _experessions.py_ : nice pieces of Theano code such as monotonicity penalty, weights entropy

### How to use it

1. Install blocks, see http://blocks.readthedocs.org/en/latest/setup.html.

2. Make sure that _$FUEL_PATH/timit_ contains all required data (I will archive it and upload 
   to your server, Jan)

3. Prepare the normalization parameters. Forcing feature means to be zero and variances 
   one has proven to be crucial to make anything work.
 
   ``firsttry/main.py init_norm norm.pkl``
 
   That will create a pickle _norm.pkl_ in the current directory.

4. Run training. Something like this should do the job:

   ``lvsr/firsttry/main.py train timit_bothgru_cumsum.pkl lvsr/configs/timit_bothgru_cumsum.py`` 

    _norm.pkl_ should be in the same directory where training is started.

5. Use the trained model:

   ``lvsr/firsttry/main.py search timit_bothgru_cumsum_model.pkl``

   Currently it needs a GPU to beam-search, but this can fixed very quickly 
   (I just have a lot of them and don't care too much).
  

