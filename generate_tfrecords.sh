#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10000M

source /project/cq-training-1/project1/teams/team09/venv/bin/activate
python /project/cq-training-1/project1/teams/team09/ift6759_project1/generate_batches.py --pickle_path /project/cq-training-1/project1/teams/team09/pickles_16bit --cfg_path /project/cq-training-1/project1/teams/team09/ift6759_project1/config_files/data_config_16b
it.json --train_datetimes_path /project/cq-training-1/project1/teams/team09/ift6759_project1/config_files/train_datetimes_43912.json --val_datetimes_path /project/cq-training-1/project1/teams/team09/ift6759_project1/config_files/valid_datetimes_1417.json
