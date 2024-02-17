#!/bin/sh

python3 training.py ../data ../models $1 > ../models/stats_$1.txt
trtexec --onnx=../models/model_$1.onnx --saveEngine=../models/model_$1.trt
../build/deep_ww -model ../models/model_$1.trt -snapshots ../data/snapshots_$1.csv -games 5000
