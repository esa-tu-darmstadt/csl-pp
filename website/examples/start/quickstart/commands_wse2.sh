#!/usr/bin/env bash

set -e

csl-pp ./src/layout.csl --arch=wse2 --fabric-dims=8,3 --fabric-offsets=4,1 \
  --params=N:8 -o out --memcpy --channels=1
cs_python ./run.py --name out
