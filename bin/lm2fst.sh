#!/usr/bin/env bash


LMFILE=$1
DIR=$2
KU=$KALDI_ROOT/egs/wsj/s5/utils

use_initial_eol=true

cat $LMFILE | \
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

disambig_symbols=`$KU/add_lex_disambig.pl $DIR/lexicon.txt $DIR/lexicon_disambig.txt`

ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST. It won't hurt even if don't use it
#start at 1, becuse we treat <spc> to be #0!!!
( for n in `seq 1 $ndisambig`; do echo '#'$n; done ) >$DIR/disambig.txt

cat $DIR/chars.txt | cut -d ' ' -f 1 | \
	#add disambiguatio symbols
	cat - $DIR/disambig.txt | \
	#alias <spc> with #0!!
	#sed -e 's/<spc>/#0/' | \
	awk '{ print $0, NR-1;}' > $DIR/chars_disambig.txt

$KU/make_lexicon_fst.pl                       \
    $DIR/lexicon_disambig.txt  |\
    fstcompile                                   \
        --isymbols=$DIR/chars_disambig.txt                 \
        --osymbols=$DIR/words.txt                       \
        --keep_isymbols=false --keep_osymbols=false |\
    fstaddselfloops  \
        "echo `grep -oP '(?<=<spc> )[0-9]+' $DIR/chars_disambig.txt` |" \
        "echo `grep -oP '(?<=#0 )[0-9]+' $DIR/words.txt` |"  | \
    fstarcsort --sort_type=ilabel > $DIR/L_disambig.fst

if `$use_initial_eol`; then
	initial_readout='<eol>'
else
	initial_readout='<eps>'
fi

{
	#emit initial space!
	echo "0 1 $initial_readout <spc>";
	#then loop through the rest of the input tape
	cat $DIR/chars_disambig.txt | grep -v '<eps>' | grep -v '<eol>' |  cut -d ' ' -f 1 | \
	while read p; do
		echo "1 1 $p $p"
	done
	#the <eol> transition to the final state
	echo "1 2 <eol> <eps>"
	#the final state
	echo "2"
} > $DIR/emit_a_space.fst

fstcompile \
	--isymbols=$DIR/chars_disambig.txt \
	--osymbols=$DIR/chars_disambig.txt \
	--keep_isymbols=false --keep_osymbols=false \
	$DIR/emit_a_space.fst | \
	fstarcsort --sort_type=olabel | \
	fsttablecompose - $DIR/L_disambig.fst | \
	fstarcsort --sort_type=olabel | \
	fsttablecompose - $DIR/G.fst         |\
	fstrmsymbols <(cat $DIR/chars_disambig.txt | grep '#' | cut -d ' ' -f 2) | \
    fstdeterminizestar --use-log=true        | \
    fstminimizeencoded > $DIR/LG.fst


fstprint -isymbols=$DIR/chars_disambig.txt -osymbols=$DIR/words.txt $DIR/LG.fst | \
	fstcompile --isymbols=$DIR/chars.txt                 \
        --osymbols=$DIR/words.txt                       \
        --keep_isymbols=true --keep_osymbols=true > $DIR/LG_withsyms.fst
