#!/usr/bin/env bash

paste -d ' '\
 <( cat $1 | grep Utterance | sed 's/.*(\(.*\))/\1/' )\
 <( cat $1 | grep Recognized: | sed 's/Recognized: \(.*\)/\1/' )

