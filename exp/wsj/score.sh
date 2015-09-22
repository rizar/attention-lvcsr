#!/bin/bash
set -e

KU=$KALDI_ROOT/egs/wsj/s5/utils
KL=$KALDI_ROOT/egs/wsj/s5/local

. $KU/parse_options.sh

if [ $# -ne 2 ]; then
	echo "usage: `basename $0` <dir> <part>"
	exit 1
fi

dir=$1
part=$2

# Aggregate groundtruth
cat $dir/$part-groundtruth-text.txt | sort | $KL/wer_ref_filter > $dir/tmp
mv $dir/tmp $dir/$part-groundtruth-text.txt

# Aggregate decoded
$LVSR/bin/decoded_chars_to_words.py $lexicon $dir/$part-decoded.out - | $KL/wer_hyp_filter > $dir/$part-decoded-text.out

# Score
compute-wer --text --mode=all ark:$dir/$part-groundtruth-characters.txt ark:$dir/$part-decoded.out $dir/$part-characters.errs > $dir/$part-characters.wer
compute-wer --text --mode=all ark:$dir/$part-groundtruth-text.txt ark:$dir/$part-decoded-text.out $dir/$part-text.errs > $dir/$part-text.wer
