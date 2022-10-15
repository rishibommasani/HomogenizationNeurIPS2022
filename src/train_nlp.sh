#!/bin/bash

for i in {1..5}
  do
    CMD="python3 train_nlp.py --config conf/train_nlp.yaml --seed $i"
    eval $CMD
  done
