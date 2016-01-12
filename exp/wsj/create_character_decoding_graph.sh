#!/usr/bin/env bash


KU=$KALDI_ROOT/egs/wsj/s5/utils

use_bol=false

. $KU/parse_options.sh

if [ $# -ne 2 ]; then
	echo "usage: `basename $0` <lm_file> <dir>"
	echo "options:"
	echo "		--use-bol (true|false)        #default: false, if true the graph will accout for bol symbol"
	exit 1
fi

LMFILE=$1
DIR=$2

WSJDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p $DIR

$WSJDIR/create_character_lexicon.sh $LMFILE $DIR
lm2fst.sh --use-bol $use_bol $LMFILE $DIR

lm_num_lines=`wc -l $LMFILE`
if [ `cat $LMFILE | wc -l` -le 50 ]; then
	fstdraw -isymbols=$DIR/words.txt -osymbols=$DIR/words.txt $DIR/G.fst | dot -Tpdf > $DIR/G.pdf
	fstdraw -isymbols=$DIR/chars_disambig.txt -osymbols=$DIR/words.txt $DIR/L_disambig.fst | dot -Tpdf > $DIR/L_disambig.pdf
	fstdraw -isymbols=$DIR/chars_disambig.txt -osymbols=$DIR/words.txt $DIR/LG.fst | dot -Tpdf > $DIR/LG.pdf
fi
