#!/usr/bin/env bash
set -uex

MODEL=$1
PART=$2
BEAM_SIZE=$3
LM=${LM:=nolm}
LM_PATH=${LM_PATH:=data/lms/wsj_trigram_no_initial_eos}

ls $MODEL/reports || mkdir $MODEL/reports

if [ $LM == nolm ]
then
    COMMON_LM_CONF="monitoring.search.char_discount 0.1"
else
    COMMON_LM_CONF=\
    "monitoring.search.beam_size $BEAM_SIZE monitoring.search.char_discount 1.0"\
    " net.lm.weight 0.5 net.lm.no_transition_cost 20"\
    " net.lm.path '$LM_PATH/LG_pushed_withsyms.fst'"
fi

$LVSR/bin/run.py search --part=$PART\
    --report $MODEL/reports/${PART}_${LM}_${BEAM_SIZE}\
    $MODEL/annealing1_best_ll.zip $LVSR/exp/wsj/configs/$MODEL.yaml\
    vocabulary $LM_PATH'/words.txt' net.prior.before 10\
    $COMMON_LM_CONF
