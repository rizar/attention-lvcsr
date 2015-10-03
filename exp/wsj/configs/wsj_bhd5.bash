#!/usr/bin/env bash
$LVSR/lvsr/run.py --num-epochs=1 train wsj_bhd5.zip $LVSR/exp/wsj/configs/wsj_bhd5.yaml 
$LVSR/lvsr/run.py --params wsj_bhd5.zip train wsj_bhd5r.zip $LVSR/exp/wsj/configs/wsj_bhd5.yaml net.prior.type "'window_around_mean'"
