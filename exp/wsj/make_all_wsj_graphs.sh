#!/bin/bash

LMFILE=$FUEL_DATA_PATH/WSJ/lm_bg.arpa.gz
WSJDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


NDIR=wsj_dict_no_initial_eos
NLM=$NDIR/lm_dict.arpa.gz
mkdir -p $NDIR
gzip -cd $LMFILE | \
	arpa_lm_to_dict_lm.py | \
	gzip -c  > $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh $NLM $NDIR

NDIR=wsj_unigram_no_initial_eos
NLM=$NDIR/lm_unigram.arpa.gz
mkdir -p $NDIR
gzip -cd $LMFILE | \
	arpa_lm_to_unigram_lm.py | \
	gzip -c  > $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh $NLM $NDIR

NDIR=wsj_bigram_no_initial_eos
NLM=$NDIR/lm_bg.arpa.gz
mkdir -p $NDIR
cp $LMFILE $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh $NLM $NDIR
