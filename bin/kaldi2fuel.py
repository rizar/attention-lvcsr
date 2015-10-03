#!/usr/bin/env python

"""
This script excnages data between a Fuel's HDF5 dataset and Kaldi's archives.
"""

import sys

import numpy
import h5py

import logging
logger = logging.getLogger(__file__)

import kaldi_io, kaldi_argparse

from fuel.datasets.hdf5 import H5PYDataset


def get_parser(datasets={}):
    parser = kaldi_argparse.KaldiArgumentParser(description="""Exchange data between Kaldi and Fuel's hdf5 dataset""", )
    parser.add_argument("h5file")
    subparsers = parser.add_subparsers(help="action")

    parser_add_data = subparsers.add_parser('add', help="add data to the hdf5 file from a Kaldi archive")
    parser_add_data.add_argument("rxfilename")
    parser_add_data.add_argument("sourcename")
    parser_add_data.add_argument("--type", default="BaseFloatMatrix",
                                 help="Kaldi reader type, the value type can be later changed via the --transform argument")
    parser_add_data.add_argument("--transform", default=None,
                                help="string whose eval()uation should produce a lambda function to porcess elements")
    parser_add_data.add_argument("--applymap", default=None,
                                help="path to file which converts data into numeric values. If a transform function is given, the data is first transformtd, then mapped")
    parser_add_data.set_defaults(func=add_data)

    parser_add_raw_text = subparsers.add_parser('add_raw_text', help="add raw text to the hdf5 file from a Kaldi text file")
    parser_add_raw_text.add_argument("textfilename")
    parser_add_raw_text.add_argument("sourcename")
    parser_add_raw_text.set_defaults(func=add_raw_text, transform=None, applymap=None)

    parser_readdata = subparsers.add_parser('read_raw_text', help="read data from the hdf5 as text")
    parser_readdata.add_argument("sourcename")
    parser_readdata.add_argument("wxfilename")
    parser_readdata.add_argument("--subset", default=None,
                                 help="Which subset to read, by default read all data")
    parser_readdata.set_defaults(func=read_raw_text)

    parser_add_text = subparsers.add_parser('add_text', help="add raw text to the hdf5 file from a Kaldi text file")
    parser_add_text.add_argument("--applymap", default=None, required=True,
                                help="path to file which converts data into numeric values. If a transform function is given, the data is first transformtd, then mapped")
    parser_add_text.add_argument("textfilename")
    parser_add_text.add_argument("sourcename")
    parser_add_text.set_defaults(func=add_text, transform=None, applymap=None)

    parser_readdata = subparsers.add_parser('read_text', help="read data from the hdf5 and convert to text")
    parser_readdata.add_argument("sourcename")
    parser_readdata.add_argument("wxfilename")
    parser_readdata.add_argument("--subset", default=None,
                                 help="Which subset to read, by default read all data")
    parser_readdata.set_defaults(func=read_text)

    parser_readdata = subparsers.add_parser('read', help="read data from the hdf5 into a kaldi archive")
    parser_readdata.add_argument("type")
    parser_readdata.add_argument("sourcename")
    parser_readdata.add_argument("rxfilename")
    parser_readdata.add_argument("--subset", default=None,
                                 help="Which subset to read, by default read all data")
    parser_readdata.add_argument("--transform", default=None,
                                help="string whose eval()uation should produce a lambda function to porcess elements")
    parser_readdata.set_defaults(func=read_data)

    parser_read_symbols = subparsers.add_parser('read_symbols', help="read a symbol table")
    parser_read_symbols.add_argument('sourcename')
    parser_read_symbols.add_argument('outfilename', default='-',
                                     help="file to which write the extracted symbol table")
    parser_read_symbols.set_defaults(func=read_symbols)

    parser_adddata = subparsers.add_parser('split', help="Write down the split table.",
                                           description="""
Provide split names along with files, whose first column is treated as list of utterance ids belonging to that split.

Note: this has to be performed after each source addition.
"""
    )
    parser_adddata.add_argument("sets", nargs="*", help="Subset definitions", default="")
    parser_adddata.set_defaults(func=add_sets)

    parser_add_attr = subparsers.add_parser(
        'add_attr', help="Add attribute to a data source")
    parser_add_attr.add_argument(
        "--type", default="BaseFloatVector",
        help="Kaldi reader type, the value type can be later changed via the "
             "--transform argument.")
    parser_add_attr.add_argument("sourcename")
    parser_add_attr.add_argument("attr")
    parser_add_attr.add_argument("rxfilename")
    parser_add_attr.set_defaults(func=add_attr)

    parser.add_standard_arguments()
    return parser


def add_from_iter(args, data_iter, peeked_val):
    """
    Add data from the data_iter iterator. Will work for 1D and 2D numpy arrays and strings.
    """
    if args.transform is None:
        T = lambda x:x
    else:
        T = eval(args.transform)

    if args.applymap is not None:
        with open(args.applymap ,'r') as mf:
            value_map = {}
            for l in mf:
                val, num = l.strip().split()
                value_map[val] = int(num)
        _oldT = T
        T = lambda x: numpy.asarray([value_map[e] for e in _oldT(x)])

    with h5py.File(args.h5file, 'a') as h5file:

        if 'uttids' in h5file:
            has_uttids = True
            uttids = h5file['uttids']
        else:
            has_uttids = False
            uttids = h5file.create_dataset("uttids", (0,),
                                           dtype=h5py.special_dtype(vlen=unicode),
                                           maxshape=(None,))
            uttids.dims[0].label = 'batch'

        if has_uttids:
            num_utts = uttids.shape[0]
            max_utts = num_utts
        else:
            num_utts = 0
            max_utts = None

        peeked_val = T(peeked_val)

        if isinstance(peeked_val, numpy.ndarray):
            shapes = h5file.create_dataset("{}_shapes".format(args.sourcename), (num_utts,peeked_val.ndim),
                                           dtype='int32',
                                           maxshape=(max_utts,peeked_val.ndim))

            shape_labels = h5file.create_dataset("{}_shape_labels".format(args.sourcename), (peeked_val.ndim,),
                                                 dtype='S7')
            shape_labels[...] = ['frame'.encode('utf8'),
                                 'feature'.encode('utf8')][:peeked_val.ndim]

            dataset = h5file.create_dataset(args.sourcename, (num_utts,),
                                            dtype=h5py.special_dtype(vlen=peeked_val.dtype),
                                            maxshape=(max_utts,))
            dataset.dims[0].label = 'batch'
            dataset.dims.create_scale(shapes, 'shapes')
            dataset.dims[0].attach_scale(shapes)

            dataset.dims.create_scale(shape_labels, 'shape_labels')
            dataset.dims[0].attach_scale(shape_labels)
        elif isinstance(peeked_val, (str, unicode)):
            dataset = h5file.create_dataset(args.sourcename, (num_utts,),
                                            dtype=h5py.special_dtype(vlen=unicode),
                                            maxshape=(max_utts,))
            dataset.dims[0].label = 'batch'
        else:
            raise Exception('Can only add numpy arrays and strings')

        if args.applymap is not None:
            value_map_arr = numpy.fromiter(value_map.iteritems(),
                                           dtype=[('key','S{}'.format(max(len(k) for k in value_map.keys()))),
                                                  ('val','int32')])
            dataset.attrs['value_map'] = value_map_arr

        for utt_num, (uttid, value) in enumerate(data_iter):
            value = T(value)

            if dataset.shape[0]<=utt_num:
                dataset.resize((utt_num+1,))

            if isinstance(value, numpy.ndarray):
                if shapes.shape[0]<=utt_num:
                    shapes.resize((utt_num+1, shapes.shape[1]))
                shapes[utt_num,:] = value.shape
                dataset[utt_num] = value.ravel()
            else:
                dataset[utt_num] = value

            if has_uttids:
                if uttids[utt_num] != uttid:
                    raise Exception("Warning, read uttid: {}, expected: {}".format(uttid, uttids[utt_num]))
            else:
                uttids.resize((utt_num+1,))
                uttids[utt_num] = uttid
        if has_uttids:
            if utt_num != uttids.shape[0]-1:
                raise Exception("Too few values provided: got {}, expected: {}".format(utt_num+1, uttids.shape[0]))


def add_attr(args):
    kaldi_reader = getattr(kaldi_io, "Sequential{}Reader".format(args.type))
    with kaldi_reader(args.rxfilename) as data_iter:
        with h5py.File(args.h5file, 'a') as h5file:
            for name, data in data_iter:
                attr_name = '{}_{}'.format(args.attr, name)
                h5file[args.sourcename].attrs[attr_name] = data


def read_data(args):
    raise NotImplementedError()


def add_data(args):
    kaldi_reader = getattr(kaldi_io, "Sequential{}Reader".format(args.type))
    with kaldi_reader(args.rxfilename) as data_iter:
        return add_from_iter(args, data_iter, data_iter._kaldi_value())


def add_raw_text(args):
    if args.textfilename == '-':
        tf = sys.stdin
    else:
        tf = open(args.textfilename)
    try:
        line_iter = iter(tf)
        first_line = next(line_iter)
        all_lines = (l.strip().split(None, 1) for g in ([first_line], line_iter) for l in g)
        uttid, rest = first_line.strip().split(None, 1)
        return add_from_iter(args, all_lines, rest)
    finally:
        if tf != sys.stdin:
            tf.close()


def add_text(args):
    if args.textfilename == '-':
        tf = sys.stdin
    else:
        tf = open(args.textfilename)
    try:
        line_iter = iter(tf)
        first_line = next(line_iter)
        all_lines = (l.strip().split(None, 1) for g in ([first_line], line_iter) for l in g)
        split_lines = ((uttid, r.split()) for (uttid,r) in all_lines)
        first_line = first_line.strip().split()
        return add_from_iter(args, split_lines, first_line[1:])
    finally:
        if tf != sys.stdin:
            tf.close()


def add_sets(args):
    with h5py.File(args.h5file, 'a') as h5file:
        sources = []
        for dataset in h5file:
            if (dataset.endswith('_indices') or dataset.endswith('_shapes') or
                dataset.endswith('_shape_labels')):
                continue
            sources.append(dataset)

        uttid2idx = {uttid:idx for (idx,uttid) in enumerate(h5file['uttids']) }

        split_dict = {}
        for subset in args.sets:
            name, uttids_fname = subset.split('=')
            idxs = []
            with open(uttids_fname) as uf:
                for l in uf:
                    uttid = l.strip().split()[0]
                    idxs.append(uttid2idx[uttid])

            indices_name = '{}_indices'.format(name)

            if indices_name in h5file:
                del h5file[indices_name]

            #
            # Note: ideally, we would sort the indeces and do:
            # h5file[indices_name] = numpy.array(sorted(idxs))
            # but this would cause incompatibility with Kaldi, which keeps utterances sorted by uttid!
            #
            h5file[indices_name] = numpy.array(idxs)
            indices_ref = h5file[indices_name].ref
            split_dict[name] = {source : (-1, -1, indices_ref) for source in sources}

        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)


def read_symbols(args):
    with h5py.File(args.h5file, 'r') as h5file:
        value_map = h5file[args.sourcename].attrs['value_map']

    if args.outfilename == '-':
        out_file = sys.stdout
    else:
        out_file=args.outfilename
    value_map.sort(order=('val',))
    numpy.savetxt(out_file, value_map, fmt="%s %d")


def get_indices(h5file, subset=None):
    if subset is None:
        return range(h5file['uttids'].shape[0])
    else:
        return h5file[subset + '_indices']


def read_raw_text(args):
    out_file = sys.stdout
    h5file = None
    try:
        if args.wxfilename != '-':
            out_file=open(args.wxfilename, 'w')

        h5file = h5py.File(args.h5file, 'r')

        indices = get_indices(h5file, args.subset)
        uttids = h5file['uttids']
        data = h5file[args.sourcename]
        for idx in indices:
            out_file.write("{} {}\n".format(uttids[idx], data[idx]))

    finally:
        if out_file != sys.stdout:
            out_file.close()
        if h5file is not None:
            h5file.close()

def read_text(args):
    h5file = None
    out_file = sys.stdout
    try:
        if args.wxfilename != '-':
            out_file=open(args.wxfilename, 'w')

        h5file = h5py.File(args.h5file, 'r')

        indices = get_indices(h5file, args.subset)
        uttids = h5file['uttids']
        data = h5file[args.sourcename]
        value_map = lambda x: x
        if 'value_map' in data.attrs:
            _map = dict((v,k) for k,v in data.attrs['value_map'])
            value_map = lambda x: _map[x]

        for idx in indices:
            chars = data[idx]
            chars = [value_map(c) for c in chars]
            out_file.write("{} {}\n".format(uttids[idx], ' '.join(chars)))

    finally:
        if out_file != sys.stdout:
            out_file.close()
        if h5file is not None:
            h5file.close()

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    args = parser.parse_args()
    args.func(args)
