#!/usr/bin/env bash

set -e

cslc ./src/layout.csl --arch wse3 --fabric-dims=12,7 --fabric-offsets=4,1 \
--params=width:5,height:5,MAX_ZDIM:5 --params=BLOCK_SIZE:2 -o=out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python ./run.py -m=5 -n=5 -k=5 --latestlink out --channels=1 \
--width-west-buf=0 --width-east-buf=0 --zDim=5 --run-only --max-ite=2
