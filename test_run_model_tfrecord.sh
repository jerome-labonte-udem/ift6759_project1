#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:k80:1
#SBATCH --mem=4G
#SBATCH --time=8:00:00

FOLDER="/project/cq-training-1/project1/teams/team09"
ZIP_FILE="tf_records.zip"

# 1. Create your environement locally
source "${FOLDER}/venv/bin/activate"

# 2. Copy your dataset on the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
cp "${FOLDER}/${ZIP_FILE}" "${SLURM_TMPDIR}"

# 3. Eventually unzip your dataset
unzip "${SLURM_TMPDIR}/${ZIP_FILE}" -d "${SLURM_TMPDIR}"

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python ${FOLDER}/ift6759_project1/train.py \
        --save_dir "${FOLDER}" \
        --cfg_path "${FOLDER}/ift6759_project1/config_files/train_config_philippe.json" \
        --data_path "${SLURM_TMPDIR}/${FOLDER}/tf_records"

# 5. Copy whatever you want to save on $SCRATCH
# cp "${SLURM_TMPDIR}/saved_models/vgg2d" ${SAVE_PATH}