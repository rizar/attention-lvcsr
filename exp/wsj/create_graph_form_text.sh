#!/usr/bin/env bash

set -e

KU=$KALDI_ROOT/egs/wsj/s5/utils
WSJDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


part=test_dev93
dataset=$FUEL_DATA_PATH/wsj.h5
use_bol=false

. $KU/parse_options.sh

if [ $# -ne 1 ]; then
	echo "usage: `basename $0` <dir>"
	echo "options:"
	echo "		--part         #default: test_dev93"
	echo '		--dataset	   #default: $FUEL_DATA_PATH/WSJ/wsj_new.h5'
	echo "		--use-bol (true|false)        #default: false, if true the graph will accout for bol symbol"
	exit 1
fi

DIR=$1
LMFILE=$DIR/lm_dict.arpa.gz

mkdir -p $DIR

$LVSR/bin/kaldi2fuel.py $dataset read_raw_text --subset $part kaldi_text $DIR/tmp_raw_text.txt
cat $DIR/tmp_raw_text.txt | sort | tr -d '*:' > $DIR/raw_text.txt

rm $DIR/tmp_raw_text.txt

$LVSR/bin/create_dict_lm_from_text.sh $DIR/raw_text.txt $LMFILE

$WSJDIR/create_character_lexicon.sh $LMFILE $DIR
$LVSR/bin/lm2fst.sh --use-bol $use_bol $LMFILE $DIR
