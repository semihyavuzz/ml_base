"""Runner module for evaluating a trained logistic regression model"""
__author__ = "semih yavuz"

import numpy as np
import argparse
from helpers.data_helper import load_data
from helpers.io_helper import load_logres_model, write_prediction_results
from models.logistic_regression import LogisticRegression


def compute_predictions(model, X, Y):
    predicted_targets = model.predict(X)
    correct_pred_cnt = np.sum(Y == predicted_targets)
    num_of_examples = X.shape[0]
    acc = (1.0 * correct_pred_cnt) / (1.0 * num_of_examples)
    print("Accuracy: {}".format(acc))
    predicted_scores = model.prob(X)
    return predicted_targets, predicted_scores


def main():
    parser = argparse.ArgumentParser("run_train_logres_model.py")
    parser.add_argument("--model_file", type=str,
                        help="Path to a file (.npz file) in which the trained model is stored")
    parser.add_argument("--features_file", type=str,
                        help="Training data features, each line is corresponding to "
                             "one example's features, separated by space")
    parser.add_argument("--targets_file", type=str,
                        help="Training data true targets, each line is corresponding "
                             "to ones example's true target, aligned with features file")
    parser.add_argument("--output_result_file", type=str,
                        help="Main output directory in which another directory to save "
                             "info for this training will be created!")
    parser.add_argument("--output_features_and_true_targets", action='store_true',
                        help="Whether to output features and true targets along with predictions")
    args = parser.parse_args()

    # load model
    logres_model = LogisticRegression()
    load_logres_model(args.model_file, logres_model)

    # load data
    X, Y = load_data(args.features_file, args.targets_file)
    predicted_targets, predicted_scores = compute_predictions(model=logres_model, X=X, Y=Y)

    # write results out
    if args.output_features_and_true_targets:
        write_prediction_results(args.output_result_file, predicted_targets, predicted_scores, X=X, Y=Y)
    else:
        write_prediction_results(args.output_result_file, predicted_targets, predicted_scores)


if __name__ == '__main__':
    main()
