#!/bin/bash

LMFILE=$1
LMSDIR=$2

WSJDIR=$LVSR/exp/wsj


NDIR=$LMSDIR/wsj_dict_no_initial_eos
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

NDIR=$LMSDIR/wsj_dict_dev93_no_initial_eos
$WSJDIR/create_graph_form_text.sh $NDIR

for fst in $NDIR/LG*.fst; do
	remove_fst_weights.py $fst
done

NDIR=$LMSDIR/wsj_unigram_no_initial_eos
NLM=$NDIR/lm_unigram.arpa.gz
mkdir -p $NDIR
gzip -cd $LMFILE | \
	arpa_lm_to_unigram_lm.py | \
	gzip -c  > $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh $NLM $NDIR

NDIR=$LMSDIR/wsj_bigram_no_initial_eos
NLM=$NDIR/lm_bg.arpa.gz
mkdir -p $NDIR
cp $LMFILE $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh $NLM $NDIR

NDIR=$LMSDIR/wsj_bigram_nondet_no_initial_eos
NLM=$NDIR/lm_bg.arpa.gz
mkdir -p $NDIR
cp $LMFILE $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh $NLM $NDIR

NDIR=$LMSDIR/wsj_bigram_nondet_with_initial_eos
NLM=$NDIR/lm_bg.arpa.gz
mkdir -p $NDIR
cp $LMFILE $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh --use-bol true $NLM $NDIR


NDIR=$LMSDIR/wsj_trigram_no_initial_eos
NLM=$NDIR/lm_tg.arpa.gz
mkdir -p $NDIR
cp $LMFILE $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh $NLM $NDIR

NDIR=$LMSDIR/wsj_dict_with_initial_eos
NLM=$NDIR/lm_dict.arpa.gz
mkdir -p $NDIR
gzip -cd $LMFILE | \
	arpa_lm_to_dict_lm.py | \
	gzip -c  > $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh --use-bol true --deterministic true $NLM $NDIR

for fst in $NDIR/LG*.fst; do
	remove_fst_weights.py $fst
done

NDIR=$LMSDIR/wsj_dict_dev93_with_initial_eos
$WSJDIR/create_graph_form_text.sh --use-bol true $NDIR

for fst in $NDIR/LG*.fst; do
	remove_fst_weights.py $fst
done


NDIR=$LMSDIR/wsj_unigram_with_initial_eos
NLM=$NDIR/lm_unigram.arpa.gz
mkdir -p $NDIR
gzip -cd $LMFILE | \
	arpa_lm_to_unigram_lm.py | \
	gzip -c  > $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh --use-bol true --deterministic true $NLM $NDIR

NDIR=$LMSDIR/wsj_bigram_with_initial_eos
NLM=$NDIR/lm_bg.arpa.gz
mkdir -p $NDIR
cp $LMFILE $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh --use-bol true --deterministic true $NLM $NDIR

NDIR=$LMSDIR/wsj_trigram_with_initial_eos
NLM=$NDIR/lm_tg.arpa.gz
mkdir -p $NDIR
cp $LMFILE $NLM

$WSJDIR/create_character_lexicon.sh $NLM $NDIR
lm2fst.sh --use-bol true $NLM $NDIR
