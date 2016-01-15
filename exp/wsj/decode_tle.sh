#!/usr/bin/env bash
set -uex

MODEL=$1
PART=$2
BEAM_SIZE=$3
LM=${LM:=nolm}
LM_PATH=${LM_PATH:=data/lms/wsj_trigram_no_bos}

ls $MODEL/reports || mkdir $MODEL/reports

if [ $LM == nolm ]
then
    LM_CONF="monitoring.search.char_discount 0.1"
else
    LM_CONF="monitoring.search.beam_size $BEAM_SIZE"
    LM_CONF+=" net.lm.weight 0.15"
    LM_CONF+=" net.lm.path '$LM_PATH/LG_pushed_withsyms.fst'"
fi

$LVSR/bin/run.py search --part=$PART\
    --report $MODEL/reports/${PART}_${LM}_${BEAM_SIZE}\
    $MODEL/annealing1_best_ll.zip $LVSR/exp/wsj/configs/$MODEL.yaml\
    vocabulary $LM_PATH'/words.txt'\
    $LM_CONF

