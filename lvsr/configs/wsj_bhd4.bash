#!/usr/bin/env bash
$LVSR/lvsr/run.py --num-epochs=1 train wsj_bhd4.zip $LVSR/lvsr/configs/wsj_bhd4.yaml 
$LVSR/lvsr/run.py --params wsj_bhd4.zip train wsj_bhd4r.zip $LVSR/lvsr/configs/wsj_bhd4.yaml net.prior.type "'window_around_mean'"
