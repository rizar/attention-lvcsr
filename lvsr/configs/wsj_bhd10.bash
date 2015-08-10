#!/usr/bin/env bash
# like jan_baseline but with smaller initial weights
$LVSR/lvsr/run.py --params wsj_bhd8.zip train wsj_bhd10r.zip $LVSR/lvsr/configs/wsj_bhd4.yaml\
    regularization.max_norm 0 net.prior.type "'window_around_median'"
