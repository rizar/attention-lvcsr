#!/usr/bin/env bash
# Assuming that BHD6 will get to the level of wsj_jan_baseline2r_best, 
# let's add one more recurrent layer
$LVSR/lvsr/run.py --num-epochs=3 train wsj_bhd7.zip $LVSR/exp/wsj/configs/wsj_bhd4.yaml\
    regularization.max_norm 0
$LVSR/lvsr/run.py --params wsj_bhd7.zip train wsj_bhd7r.zip $LVSR/exp/wsj/configs/wsj_bhd4.yaml\
    regularization.max_norm 1 net.prior.type "'window_around_median'"
