# Basic ML Tools

Package for basic ML tools for classification and clustering. 

## Getting Started

### Prerequisites

What things you need to install the software and how to install them
* [Theano](http://deeplearning.net/software/theano/)
* [Numpy](http://www.numpy.org)
* [Python](https://www.python.org)


## Models

Below are info on supported models and how to train/eval them feeding your own data.

## 1) Logistic Regression

General purpose logistic regression model for data with real-valued features.

### Components
* Model: Logistic Regression model itself is in models/logistic_regression.py
* Runners:
    - **runners/run_train_logres_model.py**: Automatically trains a logistic regression model on the training data fed.
    - **runners/run_eval_logres_model.py**: Evaluates and outputs the predictions on the test data fed using a trained Logistic Regression model specified by *model_file* argument.

### TRAINING
* Simply run **runners/run_train_logres_model.py** with desired arguments, which can be seen by --help.
* **Important arguments**:
    - **tr_features_file** (See a sample file at **data/train_features.txt**): 
        * Contains the features of a training example per line.
        * Example: 5 examples with 2 features
        
        ```
        0.147 0.162
        -0.013 -0.039
        0.055 -0.132
        0.147 0.144
        -0.022 -0.023
        ```

    - **tr_targets_file** (See a sample file at **data/train_targets.txt**):  
        * Contains the target label of a corresponding example in features file.
        * Example: Targets of 5 examples:
        
        ```
        1
        0
        0
        1
        0
        ```
        
    - **main_out_dir**:     Main output directory in which another directory will be created for this specific training run to save model checkpoints, params, etc. This directory must exist beforehand.
    
    - **early_stop**: Whether to apply early stopping based on the performance on development set
    - **patience**: Number of epochs to wait before early stopping if no progress on the development set

* **Example training run for sanity check**:
```bashscript
cd ml_base
mkdir runs
python runners/run_train_logres_model.py \
--main_out_dir runs/ \
--tr_features_file data/train_features.txt \
--tr_targets_file data/train_targets.txt \
--num_epochs 200 \
--eval_every 2 \
--early_stop \
--patience 10 \
--learning_rate 0.01
```

### EVALUATION
* Simply run **runners/run_eval_logres_model.py** with desired arguments, which can be seen by --help.
* **Important Arguments**:
    - **features_file**: Features file for test examples. It must be in the same format as the training features file used while training.
    - **targets_file**: True targets file for test examples (to report accuracy). It must be in the same format as the training targets file.
    - **output_features_and_true_targets**: Whether to include features and true targets while outputting the predictions results. To see how it changes the format of the output file by **prepending** the predictions (label and score) with **features and true targets** at each line, please see **write_prediction_results** function in **helpers/io_helper** module.
* **Example evaluation run for sanity check**:
```bashscript
python runners/run_eval_logres_model.py \
--model_file path_to_trained_logres_npz_checkpoint \
--features_file data/test_features.txt \
--targets_file data/test_targets.txt \
--output_result_file path_to_prediction_results_txt_file
```
will output
```bashscript
Accuracy: 0.87
Predictions are written into: path_to_prediction_results_txt_file
```