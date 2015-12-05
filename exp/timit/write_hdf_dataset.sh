#!/bin/bash
# This script follows the  splits from Kaldi recipe for TIMIT but it writes data into an HDF5
# file in Fuel format. We use `kaldi2fuel.py` utility to write into HDF5 file,
# run `kaldi2fuel.py --help` for details.

set -e

datasets=(train dev test)

TIMIT_DIR=/pio/data/data/timit/TIMIT/

dir=hdf5conv
h5f=$dir/timit.h5
compression="" #or --use-blosc, experimentally the difference is very small, so no use

stage=0

main_train_set=${datasets[0]}

echo "Normalizing with respect to: $main_train_set" >&2


#
# Find all waves
#

for dt in ${datasets[*]}
do
	cat data/$dt/wav.scp
done | sort | uniq > $dir/tmp_convert_all_wav.scp

num_files=`cat $dir/tmp_convert_all_wav.scp | wc -l`
num_uttids=`cat $dir/tmp_convert_all_wav.scp | cut -f 1 -d ' ' | sort | uniq | wc -l`

[ "$num_files" -eq "$num_uttids" ] || echo "Warning: you seem to have multiple files associated with the same uttid"


$LVSR/exp/timit/read_phone60_transcripts.py $TIMIT_DIR $dir

utils/filter_scp.pl $dir/tmp_convert_all_wav.scp $dir/phones60_all > $dir/phones60_nosa


$LVSR/bin/kaldi2fuel.py $h5f $compression add_text --applymap $dir/phones60.txt $dir/phones60_nosa phones60

compute-fbank-feats --use-energy=true --num-mel-bins=40 "scp:$dir/tmp_convert_all_wav.scp" 	"ark:-" | \
	add-deltas ark:- ark,scp:$dir/tmp_fbank40.ark,$dir/tmp_fbank40.scp

compute-global-cmvn-stats.py \
	"scp:utils/filter_scp.pl data/$main_train_set/wav.scp $dir/tmp_fbank40.scp|" \
	ark:$dir/tmp_fbank_dd_cmvn_stats

apply-global-cmvn.py --global-stats=ark:$dir/tmp_fbank_dd_cmvn_stats ark:$dir/tmp_fbank40.ark ark:- | \
	$LVSR/bin/kaldi2fuel.py $h5f $compression add ark:- fbank40_dd

$LVSR/bin/kaldi2fuel.py $h5f $compression add_attr fbank40_dd cmvn ark:$dir/tmp_fbank_dd_cmvn_stats

#
# Add the split information
#

for dt in ${datasets[*]}
do
	echo $dt=data/$dt/wav.scp
done | xargs $LVSR/bin/kaldi2fuel.py $h5f $compression split all=$dir/tmp_convert_all_wav.scp

python -c "if True:
	import h5py

	with h5py.File('$h5f','a') as h5file:
		h5file.attrs['h5py_interface_version'] = 'unknown'.encode('utf-8')
		h5file.attrs['fuel_convert_version'] = 'unknown'.encode('utf-8')
		h5file.attrs['fuel_convert_command'] = '$0 $@'.encode('utf-8')
"
