#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=3:00:00

FOLDER="/project/cq-training-1/project1/teams/team09"
zip -r  ${FOLDER}/tf_records.zip ${FOLDER}/tf_records
