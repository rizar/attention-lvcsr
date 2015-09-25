#!/bin/bash

#
# This script is for repository maintenance only. It checks if all
# subtrees used in the repo have their SHAs reachable from a branch
# of their respective repositories.
#


set -e

STREEMAINTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $STREEMAINTDIR/..

cur_branch=`git rev-parse --abbrev-ref HEAD`

bad_strees=()
RED='\033[0;31m'
NC='\033[0m' # No Color

for stree in libs/*; do
    stree_sha=`git log --grep "git-subtree-dir: $stree\$" -n 1 \
	| grep 'git-subtree-split:' | sed -e 's/.*git-subtree-split:\s\+//'`
    echo "The subtree $stree is on commit ${stree_sha}"
    if ! git show $stree_sha > /dev/null 2>&1; then
	bad_strees+=($stree)
	printf "${RED}Cannot find the SHA1 ${stree_sha} for subtree $stree.${NC}"
	echo "Please git fetch it and rerun this script."
    else
	already_on=`git branch --contains ${stree_sha}`
	if [ -n "${already_on}" ]; then
	    already_on=`echo ${already_on} | tr '\n' ' '`
	    echo "The non-squashed commit ${stree_sha} of subtree $stree is already in branches: ${already_on}."
	    echo "Will not create another commit for it."
	else
	    stree_name=`echo $stree | tr '/' '-'`
	    synth_branch_name="stree_${cur_branch}_${stree_name}"
	    echo "Will create branch ${synth_branch_name} for subtree $stree SHA $stree_sha"
	    git branch --force ${synth_branch_name} ${stree_sha}
	fi
    fi
    echo
done


for bad_stree in "${bad_strees[@]}"; do
    printf "${RED}There is no object matching the SHA for stree ${bad_stree}!${NC}\n"
done

if [ -n "${bad_strees}" ]; then
    exit 1;
fi

