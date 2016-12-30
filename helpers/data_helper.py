"""Helper module for preparing/formatting data"""
__author__ = "semih yavuz"

import numpy as np


def load_data(features_file, targets_file):
    X = compute_feature_matrix(features_file)
    X = np.transpose(X)
    Y = compute_target_matrix(targets_file)

    assert X.shape[0] == Y.shape[0], "Number of training examples (obtained from features) does not match" \
                                     "the number of training examples (obtained from true targets)"

    return X, Y


def compute_feature_matrix(input_file):
    features_per_example = []
    cur_features = []
    with open(input_file) as f:
        lines = f.read().splitlines()
        for line in lines:
            cur_features = []
            str_scores = line.strip().split()
            for score in str_scores:
                cur_features.append(float(score))
            features_per_example.append(cur_features)

    num_of_features = len(cur_features)
    num_of_examples = len(features_per_example)
    feature_matrix = np.zeros((num_of_features, num_of_examples), dtype=np.float32)
    for ex_id in np.arange(num_of_examples):
        for feature_id in np.arange(num_of_features):
            feature_matrix[feature_id, ex_id] = features_per_example[ex_id][feature_id]

    print("Num of examples: {}".format(num_of_examples))
    print("Num of features: {}\n".format(num_of_features))

    return feature_matrix


def compute_target_matrix(input_file):
    labels = []
    with open(input_file) as f:
        lines = f.read().splitlines()
        for line in lines:
            labels.append(float(line))

    num_of_examples = len(labels)
    target_matrix = np.zeros((num_of_examples, 1), dtype=np.float32)
    for ex_id in np.arange(num_of_examples):
        target_matrix[ex_id] = np.float32(labels[ex_id])

    return target_matrix


def compute_batches(X, Y, batch_size):
    (n, dim) = X.shape
    columns = np.arange(dim)
    shuffled_indices = np.random.permutation(n)

    batches = []
    lookup_id = 0
    while lookup_id < n:
        current_batch_indices = []
        for it in range(0, batch_size):
            if lookup_id == n:
                break
            example_index = shuffled_indices[lookup_id]
            current_batch_indices.append(example_index)
            lookup_id += 1

        X_batch = X[np.ix_(current_batch_indices, columns)]
        # print("X batch's shape is" + str(X_batch.shape))
        Y_batch = Y[np.ix_(current_batch_indices, [0])]
        # print("Y batch's shape is" + str(Y_batch.shape))
        assert X_batch.shape[0] == len(Y_batch), "Num of training examples do not match the num of true targets!"

        batches.append((X_batch, Y_batch))
        lookup_id += 1

    return batches
