#!/usr/bin/env bash
# Try to exactly replicate the fruitful BHD3 experiment
$LVSR/lvsr/run.py --num-epochs=3 train wsj_bhd6.zip $LVSR/lvsr/configs/wsj_jan_baseline.yaml 
$LVSR/lvsr/run.py --params wsj_bhd6.zip train wsj_bhd6r.zip $LVSR/lvsr/configs/wsj_jan_baseline.yaml\
   regularization.max_norm 1 net.prior.type "'window_around_median'"
