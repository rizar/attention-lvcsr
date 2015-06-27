#!/usr/bin/env bash

# The directory where the script is
export LVSR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#python modules
export PYTHONPATH=$LVSR:$LVSR/libs/blocks:$LVSR/libs/blocks-extras:$LVSR/libs/fuel:$LVSR/libs/Theano:$LVSR/libs/picklable-itertools:$PYTHONPATH
export PATH=$LVSR/libs/blocks/bin:$LVSR/libs/fuel/bin:$PATH
