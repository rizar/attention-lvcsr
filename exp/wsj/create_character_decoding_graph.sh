#!/usr/bin/env bash

LMFILE=$1
DIR=$2

WSJDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

$WSJDIR/create_character_lexicon.sh $LMFILE $DIR
lm2fst.sh $LMFILE $DIR

fstdraw -isymbols=$DIR/words.txt -osymbols=$DIR/words.txt $DIR/G.fst | dot -Tpdf > $DIR/G.pdf
fstdraw -isymbols=$DIR/chars_disambig.txt -osymbols=$DIR/words.txt $DIR/L_disambig.fst | dot -Tpdf > $DIR/L_disambig.pdf
fstdraw -isymbols=$DIR/chars_disambig.txt -osymbols=$DIR/words.txt $DIR/LG.fst | dot -Tpdf > $DIR/LG.pdf
