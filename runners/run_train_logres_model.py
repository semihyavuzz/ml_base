"""Runner module for training a Logistic Regression Model"""
__author__ = "semih yavuz"

import sys, os, time
import argparse
import numpy as np
from datetime import datetime
from models.logistic_regression import LogisticRegression
from helpers.io_helper import save_logres_model, write_as_lines
from helpers.data_helper import compute_batches, load_data


def train_logistic_regression(model, checkpoint_dir,
                              X, Y, dev_percentage,
                              learning_rate, rho, epsilon,
                              num_epochs, eval_every, batch_size,
                              early_stop, patience):
    # shuffle and split data
    (data_size, dim) = X.shape
    feature_columns = np.arange(dim)
    # shuffle
    np.random.seed(103)
    shuffle_indices = np.random.permutation(data_size)
    X_shuffled = X[np.ix_(shuffle_indices, feature_columns)]
    Y_shuffled = Y[np.ix_(shuffle_indices, [0])]
    # split
    tr_end = data_size - int(dev_percentage * float(data_size))
    tr_indices = np.arange(tr_end)
    dev_indices = np.arange(tr_end, data_size)
    X_train = X_shuffled[np.ix_(tr_indices, feature_columns)]
    X_dev = X_shuffled[np.ix_(dev_indices, feature_columns)]
    Y_train = Y_shuffled[np.ix_(tr_indices, [0])]
    Y_dev = Y_shuffled[np.ix_(dev_indices, [0])]

    # log for progress track
    tr_losses = []
    dev_accs = []
    num_examples_seen = 0
    epoch = 0
    print("Training started")
    while epoch < num_epochs:
        if epoch % eval_every == 0:
            tr_avg_loss, tr_std_loss, tr_acc = model.calculate_total_loss(X_train, Y_train)
            tr_losses.append(tr_avg_loss)
            dev_avg_loss, dev_std_loss, dev_acc = model.calculate_total_loss(X_dev, Y_dev)
            dev_accs.append(dev_acc)
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print("{0}: Progress after {1} epochs and {2} examples seen\n"
                  "Train-set: avg_loss={3}, std_loss:{4}, acc={5}\n"
                  "Dev-set  : avg_loss={6}, std_loss:{7}, acc={8}".format(time, epoch, num_examples_seen,
                                                                          tr_avg_loss, tr_std_loss, tr_acc,
                                                                          dev_avg_loss, dev_std_loss, dev_acc))


            # save model
            file_name = "LogRes-{0}-epoch{1}-{2}.npz".format(time, epoch, model.optimizer_type)
            model_dump_path = os.path.join(checkpoint_dir, file_name)
            save_logres_model(model_dump_path, model)

            # early stop
            if early_stop and len(dev_accs)>patience:
                stop = True
                for i in range(patience):
                    if dev_accs[-i] > dev_accs[-patience]:
                        stop=False
                        break
                if stop:
                    print("Early stopping: dev perf has not improved in the last {} steps".format(patience))
                    break

            # Adjust the learning rate if loss increases
            if len(tr_losses) > 1 and tr_losses[-1] > tr_losses[-2] and model.optimizer_type == "vanilla":
                learning_rate *= 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()

        # batch update
        batches = compute_batches(X_train, Y_train, batch_size)
        for (X_batch, Y_batch) in batches:
            model.batch_update(X_batch, Y_batch, learning_rate, rho, epsilon)
            num_examples_seen += batch_size

        epoch += 1


def main():
    parser = argparse.ArgumentParser("run_train_logres_model.py")
    parser.add_argument("--tr_features_file", type=str,
                        help="Training data features, each line is corresponding to "
                             "one example's features, separated by space")
    parser.add_argument("--tr_targets_file", type=str,
                        help="Training data true targets, each line is corresponding "
                             "to ones example's true target, aligned with features file")
    parser.add_argument("--main_out_dir", type=str,
                        help="Main output directory in which another directory to save "
                             "info for this training will be created!")
    parser.add_argument("--input_dim", type=str,
                        default=2,
                        help="Number of features per example")
    parser.add_argument("--optimizer_type", type=str,
                        choices=["vanilla", "adagrad", "rmsprop"], default="rmsprop",
                        help="Optimizer to use when updating model params based on gradient")
    parser.add_argument("--l2_penalty", type=float,
                        default=0.0,
                        help="Weight of l2-norm penalty term on cost")
    parser.add_argument("--num_epochs", type=int,
                        default=51,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int,
                        default=128,
                        help="Batch size")
    parser.add_argument("--eval_every", type=int,
                        default=5,
                        help="Evaluate the progress of model every this number epochs")
    parser.add_argument("--learning_rate", type=float,
                        default=0.001,
                        help="Lerning rate")
    parser.add_argument("--rho", type=float,
                        default=0.9,
                        help="Gradient moving average decay factor")
    parser.add_argument("--eps", type=float,
                        default=1e-06,
                        help="Small value added for numerical stability")
    parser.add_argument("--dev_percentage", type=float,
                        default=0.1,
                        help="Percentage of held-out dev data from entire training data")
    parser.add_argument("--early_stop", action='store_true',
                        help="Whether to apply early stop or not")
    parser.add_argument("--patience", type=int,
                        default=5,
                        help="Num of epochs to wait before early stop if no progress on the dev set")

    args = parser.parse_args()
    print("\nParameters:")
    params = []
    for (param, value) in sorted(vars(args).items()):
        param_line = "{}={}".format(param.upper(), value)
        print(param_line)
        params.append(param_line)
    print("")

    if args.early_stop and args.patience < 1:
        raise ValueError("Unexpected patience value when early stop is enabled: {0}".format(args.patience))

    # create folder for this training
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    out_dir = os.path.abspath(os.path.join(args.main_out_dir, "-".join(("LogRes", timestamp))))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Writing into {}\n".format(out_dir))
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print("Writing checkpoints in {}\n".format(out_dir))

    # write params
    params_file = os.path.join(out_dir, "params.txt")
    write_as_lines(params, params_file)
    print("Model params are written into {}\n".format(params_file))

    # training
    # load training data
    X, Y = load_data(args.tr_features_file, args.tr_targets_file)

    # init model
    _INPUT_DIM = np.int32(args.input_dim)
    _TARGET_SIZE = np.int32(1)
    _OPTIMIZER_TYPE = args.optimizer_type
    _PENALTY_WEIGHT = np.float32(args.l2_penalty)
    logres_model = LogisticRegression(input_dim=_INPUT_DIM, target_size=_TARGET_SIZE,
                                      optimizer_type=_OPTIMIZER_TYPE, penalty_weight=_PENALTY_WEIGHT)
    # init learning params
    _LEARNING_RATE = np.float32(args.learning_rate)
    _RHO = np.float32(args.rho)
    _EPSILON = np.float32(args.eps)
    _NUM_EPOCHS = np.int32(args.num_epochs)
    _EVAL_EVERY = np.int32(args.eval_every)
    _BATCH_SIZE = np.int32(args.batch_size)

    # start training
    st = time.time()
    train_logistic_regression(model=logres_model, checkpoint_dir=checkpoint_dir,
                              X=X, Y=Y, dev_percentage=args.dev_percentage,
                              learning_rate=_LEARNING_RATE, rho=_RHO, epsilon=_EPSILON,
                              num_epochs=_NUM_EPOCHS, eval_every=_EVAL_EVERY, batch_size=_BATCH_SIZE,
                              early_stop=args.early_stop, patience=args.patience)
    end = time.time()
    print("Training took {} seconds".format(end-st))

if __name__ == '__main__':
    main()
