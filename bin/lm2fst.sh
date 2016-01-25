#!/usr/bin/env bash

#
# Create a deterministic or nondeterministic characters-to-words FST from an
# ARPA LM file.
# This script produces two variants of the FST, with and without pushing
# FST weights towards the starting state. Note, that we produce only the 
# tropical semiring FSTs unlike the Kaldi recipe which uses the log
# semiring.
#

set -e

KU=$KALDI_ROOT/egs/wsj/s5/utils

use_bol=false
deterministic=false

. $KU/parse_options.sh

if [ $# -ne 2 ]; then
	echo "usage: lm2fst.sh <lm_file> <dir>"
	echo "options:"
	echo "		--use-bol (true|false)         #default: false, if true the graph will account for bol symbol"
	echo "		--deterministic (true|false)   #default: false, if true the graph is determinized at the end"
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
    grep -v '</s> </s>'   | \ arpa2fst -             | \
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
	#add disambiguation symbols
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

if `$use_bol`; then
    #Initially we used eol symbols for beginning and end of line.
	initial_readout='<bol>'
else
	initial_readout='<eps>'
fi

{
	# possibly eat <bol> (in a case if the initial readout is <bol>)
	echo "0 1 $initial_readout <eps>";

    # possibly add transition <bol>:<eps> from state 1 to 1, since utterance can
    # have multiple bos symbols at the beginning of the line
    echo "0 0 $initial_readout $initial_readout"
	#then loop through the rest of the input tape
	cat $DIR/chars.txt | grep -v '<eps>' | grep -v '<eol>' | grep -v '<bol>' |\
	    cut -d ' ' -f 1 | \
        while read p; do
            echo "1 1 $p $p"
        done

	#the <eol> transition to the final state emit a space
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

if `$deterministic`; then
    determinize="fstdeterminizestar --use-log=true"
else
    determinize="cat"
fi

fsttablecompose $DIR/eol_to_spc_bin.fst $DIR/LG_no_eol.fst | \
    $determinize | \
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
