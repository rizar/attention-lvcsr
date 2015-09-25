#!/usr/bin/env bash
set -eu

KU=$KALDI_ROOT/egs/wsj/s5/utils
KL=$KALDI_ROOT/egs/wsj/s5/local

. $KU/parse_options.sh

if [ $# -ne 2 ]; then
	echo "usage: `basename $0` <part> <dir>"
	exit 1
fi

dataset=$FUEL_DATA_PATH/wsj.h5
part=$1
dir=$2

report=$dir/report.txt
#lexicon=data/lms/wsj_dict_no_initial_eos/lexicon.txt

# Get groundtruth
$LVSR/bin/kaldi2fuel.py $dataset read_raw_text --subset $part kaldi_text $dir/tmp
cat $dir/tmp | sort | $KL/wer_ref_filter > $dir/$part-groundtruth-text.txt

# Filter decoded
$LVSR/bin/extract_for_kaldi.sh $report | sed -e 's/<noise>/<NOISE>/g' | \
	sed -e 's/ QUOTE/ "QUOTE/g' | sed -e 's/ UNQUOTE/ "UNQUOTE/g' > $dir/tmp
# Seems we don't need it
#$LVSR/bin/decoded_chars_to_words.py $lexicon $dir/tmp - |\
$KL/wer_hyp_filter $dir/tmp > $dir/$part-decoded-text.out

# Score
compute-wer --text --mode=all ark:$dir/$part-groundtruth-text.txt ark:$dir/$part-decoded-text.out $dir/$part-text.errs > $dir/$part-text.wer
cat $dir/$part-text.wer
