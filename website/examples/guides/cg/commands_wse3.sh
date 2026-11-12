#!/usr/bin/env bash

set -e

csl-pp ./src/layout.csl --arch=wse3 --fabric-dims=12,7 --fabric-offsets=4,1 \
  --params=width:5,height:5,MAX_ZDIM:5 --import-path=../../../../csl-libs \
  -o out --memcpy --channels=1
cs_python ./run.py --name out
