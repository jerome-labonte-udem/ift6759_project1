#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:k80:1
#SBATCH --mem=4G
#SBATCH --time=2:00:00

FOLDER="/project/cq-training-1/project1/teams/team09"
preds_out_path=${1}
admin_config_file=${2}
stats_output_path=${3}

# Check if config file is valid
if [ -z "${preds_out_path}" ]; then
      echo "Error: \preds_out_path argument 1 is empty"
      exit 1
fi
if [ -z "${admin_config_file}" ]; then
      echo "Error: \admin_config_file argument 2 is empty"
      exit 1
fi
if [ -z "${stats_output_path}" ]; then
      echo "Error: \stats_output_path argument 3 is empty"
      exit 1
fi

source "${FOLDER}/venv/bin/activate"

python ${FOLDER}/ift6759_project1/src/evaluator.py \
        "${preds_out_path}" \
        "${admin_config_file}" \
        -u "${FOLDER}/config_files/evaluator_config.json"\
	      -s "${stats_output_path}"
