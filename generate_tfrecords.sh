#!/bin/bash
#SBATCH --cpus-per-task=6                # Ask for 6 CPUs
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --mem=10000M

source /project/cq-training-1/project1/teams/team09/venv/bin/activate
python /project/cq-training-1/project1/teams/team09/ift6759_project1/src/preprocess_tf_record.py \
      --cfg_path /project/cq-training-1/project1/teams/team09/ift6759_project1/config_files/data_config_tfrecord.json \
      --validation
