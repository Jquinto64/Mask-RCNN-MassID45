#!/bin/bash
#SBATCH --gres=gpu:l40s:2   # request GPU(s)
#SBATCH --cpus-per-task=8   # number of CPU cores
#SBATCH --mem=20G           # memory per node
#SBATCH --array=0           # array value (for running multiple seeds, etc)
#SBATCH --time=12:00:00
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                            # Note: You must manually create output directory "slogs" 
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=mask_rcnn_lifeplan_b_512_sahi_tiled_v9_15k_iters_new_rotations

ENV_NAME=mask_rcnn
module load StdEnv/2020
module load python/3.10.2
module load cuda/11.8.0
module load opencv/4.8.0
source /home/jquinto/projects/aip-gwtaylor/jquinto/virtualenvs/$ENV_NAME/bin/activate

# Debugging outputs
pwd
python --version
pip freeze

# LazyConfig Training Script - pretrained new baseline
TILE_SIZE=512
python tools/lazyconfig_train_net.py --num-gpus 2 \
--resume \
--config-file /h/jquinto/Mask-RCNN/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py \
--exp_id ${TILE_SIZE} \
--dataset_path /h/jquinto/MaskDINO/datasets/lifeplan_${TILE_SIZE}/ \
train.output_dir=output_${TILE_SIZE}_sahi_tiled_v9 \
dataloader.train.dataset.names=lifeplan_${TILE_SIZE}_train \
dataloader.test.dataset.names=lifeplan_${TILE_SIZE}_valid \

# # LazyConfig Training Script - from scratch
# python tools/lazyconfig_train_net.py --num-gpus 2 \
# --resume \
# --config-file /h/jquinto/Mask-RCNN/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py \
# --exp_id ${TILE_SIZE} \
# --dataset_path /h/jquinto/MaskDINO/datasets/lifeplan_${TILE_SIZE}/ \
# train.output_dir=output_${TILE_SIZE}_sahi_tiled_v9_scratch \
# train.init_checkpoint="detectron2://ImageNetPretrained/torchvision/R-50.pkl" \
# dataloader.train.dataset.names=lifeplan_${TILE_SIZE}_train \
# dataloader.test.dataset.names=lifeplan_${TILE_SIZE}_valid \
