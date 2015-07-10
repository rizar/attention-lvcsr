#!/bin/bash

LMFILE=$FUEL_DATA_PATH/WSJ/lm_bg.arpa.gz
WSJDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


NDIR=wsj_lms/wsj_dict_no_initial_eos
NLM=$NDIR/lm_dict.arpa.gz
mkdir -p $NDIR
gzip -cd $LMFILE | \
	arpa_lm_to_dict_lm.py | \
	gzip -c  > $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh $NLM $NDIR

for fst in $NDIR/LG*.fst; do
	remove_fst_weights.py $fst
done

NDIR=wsj_lms/wsj_dict_dev93_no_initial_eos
$WSJDIR/create_graph_form_text.sh $NDIR

for fst in $NDIR/LG*.fst; do
	remove_fst_weights.py $fst
done

NDIR=wsj_lms/wsj_unigram_no_initial_eos
NLM=$NDIR/lm_unigram.arpa.gz
mkdir -p $NDIR
gzip -cd $LMFILE | \
	arpa_lm_to_unigram_lm.py | \
	gzip -c  > $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh $NLM $NDIR

NDIR=wsj_lms/wsj_bigram_no_initial_eos
NLM=$NDIR/lm_bg.arpa.gz
mkdir -p $NDIR
cp $LMFILE $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh $NLM $NDIR

NDIR=wsj_lms/wsj_dict_with_initial_eos
NLM=$NDIR/lm_dict.arpa.gz
mkdir -p $NDIR
gzip -cd $LMFILE | \
	arpa_lm_to_dict_lm.py | \
	gzip -c  > $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh --use-initial-eol true $NLM $NDIR

for fst in $NDIR/LG*.fst; do
	remove_fst_weights.py $fst
done

NDIR=wsj_lms/wsj_dict_dev93_with_initial_eos
$WSJDIR/create_graph_form_text.sh --use-initial-eol true $NDIR

for fst in $NDIR/LG*.fst; do
	remove_fst_weights.py $fst
done


NDIR=wsj_lms/wsj_unigram_with_initial_eos
NLM=$NDIR/lm_unigram.arpa.gz
mkdir -p $NDIR
gzip -cd $LMFILE | \
	arpa_lm_to_unigram_lm.py | \
	gzip -c  > $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh --use-initial-eol true $NLM $NDIR

NDIR=wsj_lms/wsj_bigram_with_initial_eos
NLM=$NDIR/lm_bg.arpa.gz
mkdir -p $NDIR
cp $LMFILE $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh --use-initial-eol true $NLM $NDIR
