#!/usr/bin/env bash

paste -d ' '\
 <( cat $1 | grep Utterance | sed 's/.*(\(.*\))/\1/' )\
 <( cat $1 | grep Recognized: | sed 's/Recognized: \(.*\)/\1/' )\
 | sed 's/<noise>/<NOISE>/g'\
 | sed 's/\<QUOTE\>/"QUOTE/g'\
 | sed 's/\<END-QUOTE\>/"END-QUOTE/g'\
 | sed 's/\<UNQUOTE\>/"UNQUOTE/g'

