#!/bin/bash

split=2
cam_output_dir="results/emdb/camera"
smpl_output_dir="results/emdb/smpl"
eval_input_dir="/home/gejunchen/Work/2026-1/Datasets/EMDB"

python scripts/emdb/run_cam.py --split $split --output_dir "$cam_output_dir"
python scripts/emdb/run_smpl.py --split $split --output_dir "$smpl_output_dir"
python scripts/emdb/run_eval.py --split $split --input_dir "$eval_input_dir"
