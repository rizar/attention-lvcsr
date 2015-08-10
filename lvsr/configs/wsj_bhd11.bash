#!/usr/bin/env bash
# like bhd7 and bhd8, but with smaller initial weights and no normalization
$LVSR/lvsr/run.py --num-epochs=3 train wsj_bhd11.zip $LVSR/lvsr/configs/wsj_bhd11.yaml
$LVSR/lvsr/run.py --params wsj_bhd11.zip train wsj_bhd11r.zip $LVSR/lvsr/configs/wsj_bhd11.yaml\
    net.prior.type "'window_around_median'"
