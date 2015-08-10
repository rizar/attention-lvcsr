#!/usr/bin/env bash
# like jan_baseline but with smaller initial weights
$LVSR/lvsr/run.py --num-epochs=3 train wsj_bhd9.zip $LVSR/lvsr/configs/wsj_bhd9.yaml
$LVSR/lvsr/run.py --params wsj_bhd9.zip train wsj_bhd9r.zip $LVSR/lvsr/configs/wsj_bhd9.yaml\
    net.prior.type "'window_around_median'"
