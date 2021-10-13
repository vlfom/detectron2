#!/bin/bash
#SBATCH --job-name=coco_50pct_100e
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:2,VRAM:16G
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --output=/storage/slurm/fomenko/slurm_runs/slurm-%j.out
#SBATCH --error=/storage/slurm/fomenko/slurm_runs/slurm-%j.out
#SBATCH --mail-user=ge73hov@mytum.de
#SBATCH --mail-type=ALL

module load cuda/11.2

source /usr/stud/fomenko/miniconda3/etc/profile.d/conda.sh
conda activate pytorch37

export DETECTRON2_DATASETS=/storage/slurm/fomenko/data

CONFIG_FILE=./configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ_seghead_agnostic.py

OUTPUT_DIR_PREF=/storage/slurm/fomenko/slurm_runs/csd-detectron2
OUTPUT_DIR_SUFF=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 5 ; echo '')
OUTPUT_DIR="$OUTPUT_DIR_PREF-$OUTPUT_DIR_SUFF"

python tools/lazyconfig_train_net.py \
    --config-file $CONFIG_FILE \
    --num-gpus 2 \
    OUTPUT_DIR $OUTPUT_DIR