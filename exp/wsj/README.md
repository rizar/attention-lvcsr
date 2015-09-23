To reproduce our Wall Street Journal (WSJ) experiments, please follow the instruction below:

1. Compile a Fuel-compatible dataset file in HDF5 format. The resulting file `wsj.h5` 
   should be put to $FUEL_DATA_PATH folder.

   **TODO**

2. Compile language model FST's from ARPA-format language models provided with WSJ.

   **TODO**

3. Train the model:

   `$LVSR/bin/run.py train $LVSR/lvsr/configs/wsj_paper6.yaml`

4. Decode the model on the validation and training datasets. 

   `$LVSR/exp/wsj/decode.sh wsj_paper6 <beam-size>`

    We typically use beam size 200 to get the best performance. However, with beam size 
    the scores are typically only 10\% worse and decoding is **much** faster.

5. Score the recognized transcripts:

    **TODO**
