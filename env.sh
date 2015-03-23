#!/usr/bin/env bash

echo Change '$PYTHONPATH' so that the submodules were used.

# The directory where the script is
DIR=`pwd`
BLOCKS=$DIR/blocks
THEANO=$DIR/Theano

export PYTHONPATH=$BLOCKS:$THEANO
echo New PYTHONPATH is $PYTHONPATH
