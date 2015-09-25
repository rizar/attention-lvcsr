#!/usr/bin/env bash
# like bhd8 but with no regularization at the second stage
$LVSR/lvsr/run.py --params wsj_bhd8.zip train wsj_bhd10r.zip $LVSR/exp/wsj/configs/wsj_bhd4.yaml\
    regularization.max_norm 0 net.prior.type "'window_around_median'"
