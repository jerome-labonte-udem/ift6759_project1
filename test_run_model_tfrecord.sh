#!/bin/bash
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=10:00:00                  # The job will run for 3 hours

FOLDER="/project/cq-training-1/project1/teams/team09"
# ZIP_FILE="tf_records.zip"

# 1. Create your environement locally
source "${FOLDER}/venv/bin/activate"
# module load python/3.7
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install -r requirements.txt

# 2. Copy your dataset on the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
#cp "${FOLDER}/${ZIP_FILE}" $SLURM_TMPDIR

# 3. Eventually unzip your dataset
#unzip $SLURM_TMPDIR/${ZIP_FILE} -d $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python ${FOLDER}/ift6759_project1/train.py \
        --model_path "${FOLDER}/saved_models/vgg2d" \
        --cfg_path "${FOLDER}/ift6759_project1/config_files/train_config_philippe.json" \
        --data_path "${FOLDER}/tf_records/"

# 5. Copy whatever you want to save on $SCRATCH
#cp $SLURM_TMPDIR/logs/ ${SAVE_PATH}