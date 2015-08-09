#!/usr/bin/env bash
# BHD7, but regularize at the initial stage as well
$LVSR/lvsr/run.py --num-epochs=3 train wsj_bhd8.zip $LVSR/lvsr/configs/wsj_bhd4.yaml\
    regularization.max_norm 1
$LVSR/lvsr/run.py --params wsj_bhd8.zip train wsj_bhd8r.zip $LVSR/lvsr/configs/wsj_bhd4.yaml\
    regularization.max_norm 1 net.prior.type "'window_around_median'"
