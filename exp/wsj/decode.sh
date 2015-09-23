#!/usr/bin/env bash
set -uex

MODEL=$1
PART=$2
BEAM_SIZE=$3
LM=${LM:=nolm}
LM_PATH=${LM_PATH:=data/lms/wsj_trigram_no_initial_eos}

ls $MODEL/reports || mkdir $MODEL/reports

COMMON_CMD="--char-discount=1.0"
COMMON_LM_CONF="net.lm.weight 0.5 net.lm.no_transition_cost 20"

[ $LM == nolm ] && COMMON_CMD="--char-discount=0.1"

[ $LM == nolm ] && COMMON_LM_CONF=""
if [ ! $LM == lm ]
then
    COMMON_LM_CONF="$COMMON_LM_CONF net.lm.path '$LM_PATH/LG_pushed_withsyms.fst'"
fi

$LVSR/bin/run.py search --part=$PART --beam-size=$BEAM_SIZE\
    $COMMON_CMD\
    --report $MODEL/reports/${PART}_${LM}_${BEAM_SIZE}\
    $MODEL/annealing1_best_ll.zip $LVSR/lvsr/configs/$MODEL.yaml\
    vocabulary $LM_PATH'/words.txt' net.prior.before 10\
    $COMMON_LM_CONF
