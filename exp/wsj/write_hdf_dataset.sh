#!/bin/bash

set -e

datasets=(train_si284 train_si84 dev_dt_05 dev_dt_20 test_dev93 test_eval92 test_eval93)

#datasets=(test_eval93 test_eval92)

data=data
h5f=wsj.h5
stage=0

main_train_set=${datasets[0]}

echo "Normalizing with respect to: $main_train_set" >&2

#
# kaldi_text is the text form Kaldi - it has some normalizations, more normalizations are needed for scoring
#

for ds in ${datasets[*]}
do
	cat data/$ds/text
done | sort | uniq | $LVSR/bin/kaldi2fuel.py $h5f add_raw_text - kaldi_text


#
# Add waves
#

for dt in ${datasets[*]}
do
	cat data/$dt/wav.scp
done | sort | uniq > ./tmp_convert_all_wav.scp

num_files=`cat ./tmp_convert_all_wav.scp | wc -l`
num_uttids=`cat ./tmp_convert_all_wav.scp | cut -f 1 -d ' ' | sort | uniq | wc -l`

[ "$num_files" -eq "$num_uttids" ] || echo "Warning: you seem to have multiple files associated with the same uttid"

#wav-to-matrix 'scp:./tmp_convert_all_wav.scp' 'ark:-' | \
#	$LVSR/bin/kaldi2fuel.py $h5f add --transform "lambda x: x.astype('int16')" "ark:-" wavs

#
# Character data normalization
#
# Data cleaning remarks:
# Two utterances have empty transaltion ~~
# *text* means insertion
# ` occurs two times instead of '
# < and > occure only two times aoutside of the <NOISE> phrase
#
# We: replace <NOISE> with ~
# Then we get the list of unique characters and encode space as <spc> and ~ again as <NOISE>
# Thus the two empty sequences are spelled out as <NOISE> <NOISE>
#

echo "<spc>" > tmp_chars.txt
echo "<noise>" >> tmp_chars.txt

cat $data/$main_train_set/text | cut -d' ' -f 2- | \
	sed -e 's/<NOISE>/~/g' | \
	#sed -e 's/"/''/g' | \
	tr '`' "'" | \
	tr -d -C " ~[:alpha:]'.-" | \
	tr '\n' ' ' | sed -e 's/\(.\)/\1\n/g' | \
	sort | uniq | grep -v ' ' | grep -v '~' | sort >> tmp_chars.txt

echo "<eol>" >> tmp_chars.txt

cat tmp_chars.txt | awk '{ print $0, NR-1;}' > tmp2_chars.txt

mv tmp2_chars.txt tmp_chars.txt

for subset in ${datasets[*]}
do
	cat $data/$subset/text
done | sort | uniq | cut -d' ' -f 2- | \
	sed -e 's/<NOISE>/~/g' | \
	#sed -e 's/"/''/g' | \
	tr '`' "'" | \
	tr -d -C "\n ~[:alpha:]'.-" | \
	tr ' ' '*' | \
	sed -e 's/\(.\)/\1 /g' | sed -e 's/*/<spc>/g' -e 's/~/<noise>/g' > tmp_chars_all

cat ./tmp_convert_all_wav.scp | cut -d ' ' -f 1 | paste - tmp_chars_all > tmp2_chars_all
mv tmp2_chars_all tmp_chars_all

$LVSR/bin/kaldi2fuel.py $h5f add_text --applymap tmp_chars.txt tmp_chars_all characters

compute-fbank-feats --use-energy=true --num-mel-bins=40 "scp:tmp_convert_all_wav.scp" 	"ark:-" | \
	add-deltas ark:- ark,scp:tmp_fbank40.ark,tmp_fbank40.scp

compute-global-cmvn-stats.py \
	"scp:utils/filter_scp.pl data/$main_train_set/wav.scp tmp_fbank40.scp|" \
	ark:tmp_fbank_dd_cmvn_stats

apply-global-cmvn.py --global-stats=ark:tmp_fbank_dd_cmvn_stats ark:tmp_fbank40.ark ark:- | \
	$LVSR/bin/kaldi2fuel.py $h5f add ark:- fbank_dd

for dt in ${datasets[*]}
do
	cat data/$dt/utt2spk
done | sort | uniq > ./tmp_convert_all_utt2spk
utils/utt2spk_to_spk2utt.pl ./tmp_convert_all_utt2spk > ./tmp_convert_all_spk2utt

compute-cmvn-stats --spk2utt=ark:./tmp_convert_all_spk2utt ark:tmp_fbank40.ark ark:tmp_spk_cmvn

apply-cmvn --norm-vars=true --utt2spk=ark:./tmp_convert_all_utt2spk ark:tmp_spk_cmvn \
	ark:tmp_fbank40.ark ark:- | $LVSR/bin/kaldi2fuel.py $h5f add ark:- fbank_dd_perspk

#
# Add the split information
#
#

for dt in ${datasets[*]}
do
	echo $dt=data/$dt/wav.scp
done | xargs $LVSR/bin/kaldi2fuel.py $h5f split all=tmp_convert_all_wav.scp


python -c "if True:
	import h5py

with h5py.File('$h5f','a') as h5file:
	h5file.attrs['h5py_interface_version'] = 'unknown'.encode('utf-8')
	h5file.attrs['fuel_convert_version'] = 'unknown'.encode('utf-8')
	h5file.attrs['fuel_convert_command'] = '$0 $@'.encode('utf-8')
"
