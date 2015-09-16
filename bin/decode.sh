#!/usr/bin/env bash
set -uex

MODEL=$1
PART=$2
BEAM_SIZE=$3

ls $MODEL/reports || mkdir $MODEL/reports

function decode {
    LM=$1

    COMMON_CMD="--char-discount=1.0"
    COMMON_LM_CONF="net.lm.weight 0.5 net.lm.no_transition_cost 20"

    [ $LM == nolm ] && COMMON_CMD="--char-discount=0.1"

    [ $LM == nolm ] && COMMON_LM_CONF=""
    [ $LM == trigram ] && COMMON_LM_CONF="$COMMON_LM_CONF net.lm.path 'lms/wsj_trigram_no_initial_eos/LG_pushed_withsyms.fst'"

    $LVSR/bin/run.py search --part=$PART --beam-size=$BEAM_SIZE\
        $COMMON_CMD\
        --report $MODEL/reports/${LM}_${BEAM_SIZE}\
        $MODEL/annealing1_best_ll.zip $LVSR/lvsr/configs/$MODEL.yaml\
        vocabulary 'lms/words.txt' net.prior.before 10\
        $COMMON_LM_CONF
}

decode trigram
decode nolm
