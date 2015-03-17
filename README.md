# Fully Neural LVSR

All the code is in _lvsr_. It is structured as follows:

* the _datasets_ folder contain the dataset classes. Currently only TIMIT is available.
  It expects data in the format from _/data/lisa/data/timit/readable_, that is already
  nicely packed into numpy arrays.

* the _configs_ folder contain experiment configuration in the format understood by 
  the _firsttry_

* _firsttry_ contains the do-it-all script splitted into _main.py_ and _\_\_init.py\_\__, as
  it has be to done to ensure that unpickling of the main loop object goes smoothly.

* _attenion.py_ contains different attention mechanisms tried. Warning: low code quality, 
  lots of copy-pasted code. 

* _preprocessing.py_ contains implemented preprocessings, the only is available so far is
  log\_spectrogram

* _error\_rate.py_ : Levenshtein distance and WER

* _experessions.py_ : nice pieces of Theano code such as monotonicity penalty, weights entropy
