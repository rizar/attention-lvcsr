# Wall Street journal experiment

To reproduce our Wall Street Journal (WSJ) experiments, please follow the
instructions below. All the steps should be done at the kaldi WSJ recipe
[directory](https://github.com/kaldi-asr/kaldi/tree/master/egs/wsj/s5)
(or you can add symlinks to all the files (`local`, `steps`, `utils`, etc)
as some people do). In order to perform steps 1, 2, 5 you should source
`path.sh` file from the recipe before you sourced `$LVSR/env.sh`.

### Note
Check that `$LVSR` environment variable points to this repository and
`$LD_LIBRARY_PATH` includes a path to openfst library.

## Steps
1. Compile a Fuel-compatible dataset file in HDF5 format. This step requires
   kaldi and kaldi-python.
   
   First, run prepare data part from the WSJ recipe. You'll get all
   `*.scp` files which link waves and text.
   
   Then, run
   ```
   $LVSR/exp/wsj/write_hdf_dataset.sh
   ```
   The resulting file `wsj.h5` should be put to $FUEL_DATA_PATH folder.

2. Compile language model FST's from ARPA-format language models provided with WSJ.
   This step requires kaldi.

   `$LVSR/exp/wsj/make_all_wsj_graphs.sh <lmfile> <lmsdir>`
   
   where `<lmfile>` is the arpa languge model which goes with WSJ dataset.
   (we placed it to `$FUEL_DATA_PATH/WSJ/lm_bg.arpa.gz`) and `<lmsdir>` is a
   directry to place FST language models (we use `data/lms`).
   
3. Train the model. You don't need kaldi for training and it doesn't use any
   scripts from the recipe.

   For the End-to-End Attention-based Large Vocabulary Speech Recognition use
   the `model=wsj_paper7` and for the Task Loss Estimation for Sequence 
   Prediction use `model=wsj_reward6` in the following script:
   ```
   $LVSR/bin/run.py train <model> $LVSR/exp/wsj/configs/<model>.yaml
   ```
   ```

   This will start training and save the model to the folder `<model>`.

4. Decode the model on the validation and training datasets.

   ```
   $LVSR/exp/wsj/decode.sh <model> <part> <beam-size>
   ```

   We typically use beam size 200 to get the best performance. However, with beam size
   the scores are typically only 10\% worse and decoding is **much** faster.
   You can see reports at `wsj_paper6/reports` directory. For every language
   model option, beam size and subset there are alignment images and
   `report.txt` which contains transcripts, approximate CER and WER and other
   auxiliary information.

5. Score the recognized transcripts:

    ```
    $LVSR/exp/wsj/score.sh <part> <model>/reports/valid_trigram_200/
    ```
    Where `<part>` is `test-dev93` for the development set and `test-eval92`
    for the test set.
    
    This script produces `<part>-text.wer` and `<part>-text.errs` files in the
    corresponding directory. The first one contains WER and the second one
    the report of `compute-wer` kadli script.
