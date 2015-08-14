#!/usr/bin/env bash

set -e

KU=$KALDI_ROOT/egs/wsj/s5/utils

use_initial_eol=false

. $KU/parse_options.sh

if [ $# -ne 2 ]; then
	echo "usage: lm2fst.sh <lm_file> <dir>"
	echo "options:"
	echo "		--use-initial-eol (true|false)        #default: false, if true the graph will accout for initial eol symbol"
	exit 1
fi

LMFILE=$1
DIR=$2

if [[ $LMFILE = *.gz ]]; then
	cat_cmd="gzip -d -c"
else
	cat_cmd="cat"
fi

$cat_cmd $LMFILE | \
    grep -v '<s> <s>'   | \
    grep -v '</s> <s>'   | \
    grep -v '</s> </s>'   | \
    arpa2fst -             | \
    fstprint                | \
    $KU/eps2disambig.pl      | \
    $KU/s2eps.pl              | \
    fstcompile                   \
        --isymbols=$DIR/words.txt      \
        --osymbols=$DIR/words.txt       \
        --keep_isymbols=false       \
        --keep_osymbols=false      | \
    fstrmepsilon                    | \
    fstarcsort --sort_type=ilabel      \
    > $DIR/G.fst

ndisambig=`$KU/add_lex_disambig.pl $DIR/lexicon.txt $DIR/lexicon_disambig.txt`

ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST. It won't hurt even if don't use it
( for n in `seq 0 $ndisambig`; do echo '#'$n; done ) >$DIR/disambig.txt

cat $DIR/chars.txt | cut -d ' ' -f 1 | \
	#add disambiguatio symbols
	cat - $DIR/disambig.txt | \
	awk '{ print $0, NR-1;}' > $DIR/chars_disambig.txt

$KU/make_lexicon_fst.pl                       \
    $DIR/lexicon_disambig.txt  |\
    fstcompile                                   \
        --isymbols=$DIR/chars_disambig.txt                 \
        --osymbols=$DIR/words.txt                       \
        --keep_isymbols=false --keep_osymbols=false |\
    fstaddselfloops  \
        "echo `grep -oP '(?<=#0 )[0-9]+' $DIR/chars_disambig.txt` |" \
        "echo `grep -oP '(?<=#0 )[0-9]+' $DIR/words.txt` |"  | \
    fstarcsort --sort_type=olabel > $DIR/L_disambig.fst

fsttablecompose $DIR/L_disambig.fst $DIR/G.fst | \
	fstdeterminizestar --use-log=true        | \
	fstrmsymbols <(cat $DIR/chars_disambig.txt | grep '#' | cut -d ' ' -f 2) | \
    fstrmepslocal | \
    fstminimizeencoded | \
    fstarcsort --sort_type=ilabel | \
    cat	> $DIR/LG_no_eol.fst

if `$use_initial_eol`; then
	initial_readout='<eol>'
else
	initial_readout='<eps>'
fi

{
	#possibly eat initial <eol>
	echo "0 1 $initial_readout <eps>";
	#then loop through the rest of the input tape
	cat $DIR/chars.txt | grep -v '<eps>' | grep -v '<eol>' |  cut -d ' ' -f 1 | \
	while read p; do
		echo "1 1 $p $p"
	done
	#the <eol> transition to the final state will meit a space
	echo "1 2 <eol> <spc>"
	#the final state
	echo "2"
} > $DIR/eol_to_spc.fst

fstcompile \
	--isymbols=$DIR/chars_disambig.txt \
	--osymbols=$DIR/chars_disambig.txt \
	--keep_isymbols=false --keep_osymbols=false \
	$DIR/eol_to_spc.fst | \
	fstarcsort --sort_type=olabel | \
	cat	> $DIR/eol_to_spc_bin.fst

fsttablecompose $DIR/eol_to_spc_bin.fst $DIR/LG_no_eol.fst | \
	# fstminimizeencoded | \
	# fstdeterminizestar --use-log=true        | \
    fstminimizeencoded > $DIR/LG.fst

fstpush --push_weights=true $DIR/LG.fst | \
    fstrmepsilon > $DIR/LG_pushed.fst

fstprint -isymbols=$DIR/chars_disambig.txt -osymbols=$DIR/words.txt $DIR/LG.fst | \
	fstcompile --isymbols=$DIR/chars.txt                 \
        --osymbols=$DIR/words.txt                       \
        --keep_isymbols=true --keep_osymbols=true > $DIR/LG_withsyms.fst

fstprint -isymbols=$DIR/chars_disambig.txt -osymbols=$DIR/words.txt $DIR/LG_pushed.fst | \
	fstcompile --isymbols=$DIR/chars.txt                 \
        --osymbols=$DIR/words.txt                       \
        --keep_isymbols=true --keep_osymbols=true > $DIR/LG_pushed_withsyms.fst


fstprint $DIR/LG_withsyms.fst | \
	fstcompile --isymbols=$DIR/chars.txt --osymbols=$DIR/words.txt \
		--keep_isymbols=true --keep_osymbols=true --arc_type=log \
	> $DIR/LG_log_withsyms.fst
fstpush --push_weights=true $DIR/LG_log_withsyms.fst \
	$DIR/LG_log_pushed_withsyms.fst

