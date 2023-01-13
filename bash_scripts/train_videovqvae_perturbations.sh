#!/bin/bash 

#SBATCH --job-name=acmabl_wo_translation
#SBATCH --mem-per-cpu=2048
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:4
#SBATCH --mincpus=38
#SBATCH --nodes=1
#SBATCH --time 10-00:00:00
#SBATCH --signal=B:HUP@600
#SBATCH -w gnode061

cd /ssd_scratch/cvit/aditya1/acm_rebuttal/video_vqvae/VQVAE2-Refact

source /home2/aditya1/miniconda3/bin/activate base

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 \
python train_vqvae_perceptual.py --epoch 1000 \
--colorjit const --batch_size 1 \
--sample_folder /ssd_scratch/cvit/aditya1/video_vqvae2_results/ablation_translation_disabled_samples \
--checkpoint_dir /ssd_scratch/cvit/aditya1/video_vqvae2_results/ablation_translation_disabled_checkpoints