"""Input/Output helper functions"""
__author__ = "semih yavuz"
import numpy as np
import codecs


def write_as_lines(lines, output_file):
    with codecs.open(output_file, "w", encoding="utf-8") as out_writer:
        prefix = ""
        for line in lines:
            out_writer.write(prefix)
            prefix = "\n"
            out_writer.write(line)
    out_writer.close()


def read_as_lines(input_file):
    lines = []
    with codecs.open(input_file, "r", encoding="utf-8") as in_reader:
        text = in_reader.read()
        for line in text.split("\n"):
            lines.append(line)

    return lines


def write_prediction_results(prediction_results_file, predicted_targets, predicted_scores, X=None, Y=None):
    assert predicted_targets.shape[0] == predicted_scores.shape[0], "Shapes of targets and scores do not match!"
    num_features = 0
    if (X is not None) and (Y is not None):
        print("Data features and true target are being written along with predictions!")
        num_features = X.shape[1]

    (n, m) = predicted_scores.shape
    lines = []
    for i in range(n):
        cur_line_tokens = []
        if (X is not None) and (Y is not None):
            for j in range(num_features):
                cur_line_tokens.append(str(X[i, j]))
            cur_line_tokens.append(str(Y[i, 0]))

        cur_line_tokens.append(str(predicted_targets[i, 0]))
        cur_line_tokens.append(str(predicted_scores[i, 0]))
        lines.append(" ".join(cur_line_tokens))

    write_as_lines(lines, prediction_results_file)
    print("Predictions are written into: {}".format(prediction_results_file))


def save_logres_model(output_file, model):
    """Dumps logistic regression model parameters into an .npz file

    Args:
        output_file: path to a file in which the model will be dumped
        model: LogisticRegression object to be saved

    Returns:
        None
    """
    input_dim = model.input_dim
    target_size = model.target_size
    optimizer_type = model.optimizer_type
    penalty_weight = model.penalty_weight
    w = model.w.get_value()
    b = model.b.get_value()
    np.savez(
        output_file,
        input_dim=input_dim, target_size=target_size,
        optimizer_type=optimizer_type, penalty_weight=penalty_weight,
        w=w, b=b
    )
    print("Saved LogRes model to %s." % output_file)


def load_logres_model(input_file, model):
    """Loads a logistic regression model from a specified input file

    Args:
        input_file: path to a file from which the model will be loaded

    Returns:
        model: loaded LogisticRegression object
    """
    npzfile = np.load(input_file)
    input_dim = npzfile["input_dim"]
    target_size = npzfile["target_size"]
    optimizer_type = npzfile["optimizer_type"]
    penalty_weight = npzfile["penalty_weight"]
    w = npzfile["w"]
    b = npzfile["b"]

    # load it in model
    model.input_dim = input_dim
    model.target_size = target_size
    model.optimizer_type = optimizer_type
    model.penalty_weight = penalty_weight
    model.w.set_value(w)
    model.b.set_value(b)

    print("Loaded logistic regression model from %s." % input_file)
    return model