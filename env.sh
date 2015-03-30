#!/usr/bin/env bash

echo Change '$PYTHONPATH' so that the submodules were used.

# The directory where the script is
export LVSR=`pwd`
BLOCKS=$LVSR/blocks
THEANO=$LVSR/Theano

export PYTHONPATH=$LVSR:$BLOCKS:$THEANO
echo New PYTHONPATH is $PYTHONPATH
