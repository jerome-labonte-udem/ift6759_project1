# ift6759_project1

## Training a model:

```bash
python train.py --model_path saved_models/rnn --cfg_path config_files/train_config_jerome_rnn.json -ppath/to/save/model path/to/configfile.json -p
```
--model_path is the path where the model weights will be saved.
--cfg_path is the path to the train_config path (see examples)

The -p or --plot option is used to plot the losses after the training if 
the execution environment allows it.

Notes: 
 
 * in the config file, the script assumes that the model Class is in a module of the same name, in the package models.
For example, if model name is "RNN", we expect to find a RNN class in the models.RNN module.

* The CNN2D model only takes on image. The previous_time_offsets list in the config file should only
contain one entry (probably "P0DT0H0M0S" to get image at t0)

## Tensorboard
After running a model, you can visualize the training/validation curves according to the hyperparameters 
chosen by running the command
```bash
tensorboard --logdir logs/fit
```