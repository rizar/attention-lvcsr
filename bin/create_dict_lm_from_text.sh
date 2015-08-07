#!/usr/bin/env bash

set -e

KU=$KALDI_ROOT/egs/wsj/s5/utils

use_initial_eol=false

. $KU/parse_options.sh

if [ $# -ne 2 ]; then
	echo "usage: $0 <TEXTFILE> <LMFILE>"
	echo "options:"
	exit 1
fi

TEXTFILE=$1
LMFILE=$2

tmpfile=`mktemp`

cat $TEXTFILE | cut -d' ' -f2- | tr ' ' '\n' | sort | uniq | \
	grep -v "<UNK>" > $tmpfile

{
	echo "\\data\\"
	echo "ngram 1=`cat $tmpfile | wc -l`"
	echo "\\1-grams:"
	echo "0 <UNK>"
	echo "0 </s>"
	echo "0 <s>"
	cat $tmpfile | sed -e "s/^/0 /"
	echo "\\end\\"
} | gzip -c > $LMFILE
