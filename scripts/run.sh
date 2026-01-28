#!/bin/bash

split=2
cam_output_dir="results/emdb/camera"
smpl_output_dir="results/emdb/smpl"
eval_input_dir="/home/gejunchen/Work/2026-1/Datasets/EMDB"

python scripts/emdb_eval/run_cam.py --split $split --output_dir "$cam_output_dir"
python scripts/emdb/run_smpl.py --split $split --output_dir "$smpl_output_dir"
python scripts/emdb/run_eval.py --split $split --input_dir "$eval_input_dir"

# emdb2:
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P0/09_outdoor_walk
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P2/19_indoor_walk_off_mvs
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P2/20_outdoor_walk
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P2/24_outdoor_long_walk
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P3/27_indoor_walk_off_mvs
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P3/28_outdoor_walk_lunges
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P3/29_outdoor_stairs_up
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P3/30_outdoor_stairs_down
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P4/35_indoor_walk
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P4/36_outdoor_long_walk
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P4/37_outdoor_run_circle
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P5/40_indoor_walk_big_circle
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P6/48_outdoor_walk_downhill
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P6/49_outdoor_big_stairs_down
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P7/55_outdoor_walk
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P7/56_outdoor_stairs_up_down
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P7/57_outdoor_rock_chair
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P7/58_outdoor_parcours
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P7/61_outdoor_sit_lie_walk
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P8/64_outdoor_skateboard
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P8/65_outdoor_walk_straight
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P9/77_outdoor_stairs_up
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P9/78_outdoor_stairs_up_down
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P9/79_outdoor_walk_rectangle
# Adding root:  /home/gejunchen/Work/2026-1/Datasets/EMDB/P9/80_outdoor_walk_big_circle