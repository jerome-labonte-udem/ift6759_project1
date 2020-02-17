#!/bin/bash
#SBATCH --cpus-per-task=6                # Ask for 6 CPUs
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --mem=10000M
#SBATCH --output=./logfile

FOLDER="/project/cq-training-1/project1/teams/team09"
source ${FOLDER}/venv/bin/activate
python ${FOLDER}/ift6759_project1/src/preprocess_tf_record.py \
      --cfg_path ${FOLDER}/ift6759_project1/config_files/data_config_tfrecord.json \
      --validation
