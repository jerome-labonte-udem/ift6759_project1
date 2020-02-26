# ift6759_project1

## Run the evaluator.py on the test set
sbatch scripts/evaluator.sh [preds_output_path] [admin_config_path] [stats_output_path]
stats_output_path is optional, the first two arguments follow the order of the evaluator.py script

## Generate the training and validation sets
We used TFRecords to store our (x, y) samples where x is a series of pre-cropped past images + present image 
and their corresponding metadata, and y is the GHI values at [T0, T+1, T+3, T+6].
The training set is predefined to be the years 2010-2014 and the validation set to be 2015. 
The first step is to create a config json file that has to contain the path to the data, the save path, the patch size,
the previous time offsets, etc. See `config_files/data_config_tf_record.json` for an example. 
To start with the preprocessing:
```bash
python preprocess_tf_record.py --cfg_path config_files/data_config_tfrecord.json
```

## Training a model:
```bash
python train.py --save_dir saved_models/rnn --cfg_path config_files/train_config_jerome_rnn.json 
--data_path data/tf_records/ -p
```
--save_dir is the path to the directory where the model weights and log files will be saved. <br/>
--cfg_path is the path to the train_config path (see examples)  <br/>
--data_path is the path to the directory of the training and validation data in forms of TFRecord. 
The train data has to be in a directory called "train" (in the data_path)  and the validation data in "validation". <br/>
-p or --plot option is used to plot the losses after the training if the execution environment allows it. <br/>

Notes: 
 
 * in the config file, the script assumes that the model Class is in a module of the same name, in the package models.
For example, if model name is "RNN", we expect to find a RNN class in the models.RNN module.

* The CNN2D model only takes on image. The previous_time_offsets list in the config file should only
contain one entry (probably "P0DT0H0M0S" to get image at t0)

## Tensorboard
After running a model, you can visualize the training/validation curves according to the hyperparameters 
chosen by running the command:
```bash
tensorboard --logdir logs/fit
```

## Visualize predictions
### Get Predictions
Once the model is trained, you can run the command
```bash
python -m src.evaluator preds_file valid_config_file -u path_user_config
```
preds_file is the path of output file where you want the results to be stored. <br/>
valid_config_file is the path to a config file that contains path to the catalog.pkl, 
the list of datetimes to test, etc. See `tests/valid_cfg/valid_cfg_week_january_viz.json` for an example. <br/>
-u See `config_files/evaluator.config.json` to get an example of the fields necessary for the user config file 
(model name, previous time offsets, etc.). <br/>
### Visualize predictions v.s. targets
Last command will write all predictions to the `preds_file` and print the RMSE per station. 
You can then visualize and compare the predictions
to the clearsky model and true GHI values (per day), by running:
```bash
python -m src.utils.visualization_utils preds_file path_dataframe -t path_user_config
```
path_dataframe is the path to the pandas dataframe (here `data/catalog.helios.public.20100101-20160101.pkl`)
-t is the path to the user config file (same .json as for the src.evaluator command).
### Visualize bar plots
With a working model, you can also generate bar plots to compare the results by station,
by the cloudiness factor, and by the month of the year. The command is:
```bash
python generate_bar_plots.py cfg_path path_dataframe ---saved_predictions [path save file]
```
cfg_path is the path to the config file that has to follow the format of 
`config_files/viz_config.json`.

## Repo Structure
Executable scripts that were called directly from the bash files to run on the cluster can be found in the main directory.
All the other relevant can be found `src` folder. The `src/deprecated/` folder contains code of things that were
tried but were not part of the final implementation (e.g. Preprocessing the training data into pickle files).
The `models` folder contains the main machine learning models
that we implemented and that are compared in our report. The `data` folder contains the `catalog.pkl` as well as 
a subset of the project's data that was mainly used to test stuff locally, and to have unit tests for the data loading/preprocessing.
The `scripts` folder contains the important bash scripts that we used to run our code (preprocessing, training, visualisation)
on Mila's cluster. Finally, the `config_files` folder contains all the different configurations files
that are needed to run the various scripts of the whole machine learning pipeline (preprocess, train, test, visualize).


## ClearSky Baseline
The `src/baseline.py` allows to compute directly the RMSE per station and globally of the ClearSky model,
which gives an idea of what should be an easy baseline to beat.

