#!/usr/bin/env bash
set -e

# like bhd7 and bhd8, but with smaller initial weights and no normalization
$LVSR/lvsr/run.py --num-epochs=3 train wsj_jan_bhd05.zip $LVSR/lvsr/configs/wsj_jan_bhd05.yaml net.prior.type "'expanding'"
$LVSR/lvsr/run.py --params wsj_jan_bhd05.zip train wsj_jan_bhd05r.zip $LVSR/lvsr/configs/wsj_jan_bhd05.yaml
