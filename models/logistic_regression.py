"""Logistic Regression in Theano"""
__author__ = "semih yavuz"

import theano
import theano.tensor as T
import numpy as np


class LogisticRegression:
    def __init__(self, input_dim=2, target_size=1, optimizer_type="vanilla", penalty_weight=0.01):
        self.input_dim = input_dim
        self.target_size = target_size
        self.optimizer_type = optimizer_type
        self.penalty_weight = penalty_weight

        # model parameters
        w = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (input_dim, target_size))
        b = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (target_size, 1))
        # create shared variables for them
        self.w = theano.shared(name='w', value=w.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX), broadcastable=(True, False))

        # grad history
        w_grad_cache = np.zeros((input_dim, target_size), dtype=np.float32)
        b_grad_cache = np.zeros((target_size, 1), dtype=np.float32)
        # make these shared variables as they will be updated
        self.w_grad_cache = theano.shared(name="w_grad_cache", value=w_grad_cache.astype(theano.config.floatX))
        self.b_grad_cache = theano.shared(name="b_grad_cache", value=b_grad_cache.astype(theano.config.floatX),
                                          broadcastable=(True, False))

        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        x = T.fmatrix('x')   # batch_size * input_dim
        y = T.fmatrix('y')   # batch_size * 1

        # Construct the computation graph
        p_1 = 1 / (1 + T.exp(-T.dot(x, self.w) - self.b))
        xent_total = -y * T.log(p_1) - (1-y) * T.log(1-p_1)
        xent = T.mean(xent_total)
        cost = xent + self.penalty_weight * T.sum(self.w ** 2)
        gw, gb = T.grad(cost, [self.w, self.b])
        prediction = p_1 > 0.5

        # Update model params
        learning_rate = T.fscalar("lr")
        rho = T.fscalar("rho")
        eps = T.fscalar("epsilon")
        grad_updates = []
        if self.optimizer_type == "vanilla":
            grad_updates.append((self.w, self.w - learning_rate * gw))
            grad_updates.append((self.b, self.b - learning_rate * gb))
        elif self.optimizer_type == "adagrad":
            # update on w
            w_cache_updated = self.w_grad_cache + gw ** 2
            w_grad_scaling = T.sqrt(w_cache_updated + eps)
            gw = gw / w_grad_scaling
            grad_updates.append((self.w, self.w - learning_rate * gw))
            grad_updates.append((self.w_grad_cache, w_cache_updated))
            # update on b
            b_cache_updated = self.b_grad_cache + gb ** 2
            b_grad_scaling = T.sqrt(b_cache_updated + eps)
            gb = gb / b_grad_scaling
            grad_updates.append((self.b, self.b - learning_rate * gb))
            grad_updates.append((self.b_grad_cache, b_cache_updated))
        elif self.optimizer_type == "rmsprop":
            # update on w
            w_cache_updated = rho * self.w_grad_cache + (1. - rho) * (gw ** 2)
            w_grad_scaling = T.sqrt(w_cache_updated + eps)
            gw = gw / w_grad_scaling
            grad_updates.append((self.w, self.w - learning_rate * gw))
            grad_updates.append((self.w_grad_cache, w_cache_updated))
            # update on b
            b_cache_updated = rho * self.b_grad_cache + (1 - rho) * (gb ** 2)
            b_grad_scaling = T.sqrt(b_cache_updated + eps)
            gb = gb / b_grad_scaling
            grad_updates.append((self.b, self.b - learning_rate * gb))
            grad_updates.append((self.b_grad_cache, b_cache_updated))
        else:
            raise ValueError("Unexpected optimizer type: %s" % self.optimizer_type)

        # Compile
        # Compile
        self.batch_update = theano.function(inputs=[x, y, learning_rate, rho, eps],
                                            outputs=[xent],
                                            updates=grad_updates,
                                            on_unused_input='warn')
        self.prob = theano.function(inputs=[x], outputs=p_1)
        self.predict = theano.function(inputs=[x], outputs=prediction)
        self.error = theano.function(inputs=[x, y], outputs=xent)

    # TODO: Do it as a single batch! (or at least in larger batches)
    def calculate_total_loss(self, X, Y):
        (n, dim) = X.shape
        columns = np.arange(dim)

        loss_per_example = np.zeros(n)
        correct_pred_cnt = 0
        for row in np.arange(n):
            x = X[np.ix_([row], columns)]
            # print("x's shape is" + str(x.shape))
            y = Y[np.ix_([row], [0])]
            # print("y's shape is" + str(y.shape))
            loss_per_example[row] = self.error(x, y)
            if  self.predict(x) == y[0, 0]:
                correct_pred_cnt += 1

        avg_loss = np.mean(loss_per_example)
        std_loss = np.std(loss_per_example)
        acc = (1.0 * correct_pred_cnt) / (1.0 * n)

        return avg_loss, std_loss, acc
