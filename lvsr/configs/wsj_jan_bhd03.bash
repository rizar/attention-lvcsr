#!/usr/bin/env bash
# like bhd7 and bhd8, but with smaller initial weights and no normalization
$LVSR/lvsr/run.py --num-epochs=3 train wsj_jan_bhd03.zip $LVSR/lvsr/configs/wsj_jan_bhd03.yaml
$LVSR/lvsr/run.py --params wsj_jan_bhd03.zip train wsj_jan_bhd03r.zip $LVSR/lvsr/configs/wsj_jan_bhd03.yaml\
    net.prior.type "'window_around_median'"
