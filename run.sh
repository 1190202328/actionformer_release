#!/bin/bash

## THUMOS14 with I3D features
python ./train.py ./configs/thumos_i3d.yaml --output reproduce
python ./eval.py ./configs/thumos_i3d.yaml ./ckpt/thumos_i3d_reproduce

#python ./train.py ./configs/anet_tsp.yaml --output reproduce
#python ./eval.py ./configs/anet_tsp.yaml ./ckpt/anet_tsp_reproduce
