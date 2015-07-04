#!/usr/bin/env bash

set -e

#TODO: add command line parsing

LMFILE=$1
DIR=$2
DATASET=$FUEL_DATA_PATH/wsj/wsj_new.h5


if [[ $LMFILE = *.gz ]]; then
	cat_cmd="gzip -d -c"
else
	cat_cmd="cat"
fi


#
# WARNING:
# please note: through the script we will alias the silence character with the #0 character.
#

#extract the character table used by the net
kaldi2fuel.py $DATASET read-symbols characters $DIR/net-chars.txt


# OpenFST needs 0 to refer to the special <eps> symbol, prepend and renumber!!
{
	echo "<eps>";
	cat $DIR/net-chars.txt | cut -d ' ' -f 1;
} | awk '{ print $0, NR-1;}' > $DIR/chars.txt


#Get the word list from the LM
{
	echo "<eps>";
	$cat_cmd $LMFILE | \
	#skip up to \data\
	#then skip up to \1-grams
	#then print up to \2-grams or \end
	#then delete empty lines
		sed -e '0,/^\\data\\/d' \
			-e '0,/^\\1-grams:/d' \
			-e '/\(^\\2-grams:\)\|\(^\\end\\\)/,$d' \
			-e '/^\s*$/d' | \
	#print just the word
		awk '{print $2; }' | \
	#finally remove <s> and </s>, they will be added later
		grep -v '</\?s>'
	echo "#0";
	echo "<s>";
	echo "</s>";
} | awk '{ print $0, NR-1;}' > $DIR/words.txt

#Build the lexicon:
# establish allowed character set by looking at the net-chars.txt
# filter any unrecognized character

allowed_characters=`cat $DIR/net-chars.txt  | grep -v '<.*>' | cut -d ' ' -f 1 | tr -d '\n'`


#TODO: shall we also output noise instead of unk??
echo "<UNK>" "<noise>" > $DIR/lexicon.txt

cat $DIR/words.txt | cut -d ' ' -f 1 | \
	grep -v "<.*>" | \
	grep -v "#0" > $DIR/tmp-words-to-convert.txt

cat $DIR/tmp-words-to-convert.txt | \
	tr -c -d "\n$allowed_characters" | sed -e "s/\(.\)/ \1/g" |\
	paste -d '' $DIR/tmp-words-to-convert.txt - >> $DIR/lexicon.txt

rm $DIR/tmp-words-to-convert.txt
