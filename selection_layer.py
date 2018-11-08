"""
The selection layer used in the DeepINSIGHT model
"""

from __future__ import print_function

from itertools import islice

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import copy

import numpy
from numpy import genfromtxt

import theano
import theano.tensor as T

class SelectionLayer(object):
    def __init__(self, input, n_in, n_out, class_prob):
        """ Initialize the selection layer
        """

        # The weights of features
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out - 1),
                dtype=theano.config.floatX
            ),
            name="W",
            borrow=True
        )

        # The biases of features
        if class_prob is None:
            self.b = theano.shared(
                value=numpy.zeros(
                    n_out - 1,
                    dtype=theano.config.floatX
                ),
                name="b",
                borrow=True
            )
        else:
            if (len(class_prob) != n_out):
                raise Exception("Unmatched class probability!")

            class_prob = numpy.asarray(class_prob)
            class_prob = numpy.divide(class_prob, class_prob[-1], dtype=theano.config.floatX)
            class_prob = numpy.log(numpy.delete(class_prob, -1))
            self.b = theano.shared(
                value=class_prob,
                name="b",
                borrow=True
            )


        # dummy bias and weight
        dummy_b = theano.shared(numpy.zeros(1, dtype=theano.config.floatX), borrow=True)
        dummy_W = theano.shared(
            value=numpy.zeros(
                (n_in, 1),
                dtype=theano.config.floatX
            ),
            name="W",
            borrow=True
        )

        # the acutal bias and weight
        real_b = T.concatenate([self.b, dummy_b])
        real_W = T.concatenate([self.W, dummy_W], 1)

        # reshape input to 2D tensor
        shape = input.shape
        input_2d_shape = input.reshape((shape[0] * shape[1], shape[2]))

        # The probability matrix of selection categories
        self.selection_weight = T.nnet.softmax(T.dot(input_2d_shape, real_W) + real_b)

        # parameters
        self.params = [self.W, self.b]

        # keep track of input data
        self.input = input

    def negative_log_likelihood(self, y):
        # resahpe y to 2D tensor
        shape = y.shape
        y_2d_shape = y.reshape((shape[0] * shape[1], shape[2]))

        prob = T.sum(self.selection_weight * y_2d_shape, axis=1)

        # reshape again
        prob_reshaped = prob.reshape((-1, 3))

        # the log likelihood
        return -T.mean(T.log(T.sum(prob_reshaped, axis=1) / 3.))

    def prediction(self):
        return self.selection_weight


def RMSprop(cost, params, learning_rate, rho=0.9, epsilon=1e-6):
    """
    The RMSprop algorithm proposed by Jeffrey Hinton
    """
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - learning_rate * g))
    return updates

def AdaGrad(cost, params, learning_rate, epsilon=1e-6):
    """
    The AdaGrad algorithm. Note that it is similar to RMSprop except
    the definition of acc_new
    """
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = acc + g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - learning_rate * g))
    return updates

def stochastic_gradient_descent(cost, params, learning_rate):
    """
    The simplest SGD
    """
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append((p, p - learning_rate * g))
    return updates


def sgd_optimize_single_layer(learning_rate, feature_file, validation_file,
                              L1_reg=1.e-3, batch_size=100, nb_epoch=100):
    x = T.tensor3("x")
    y = T.tensor3("y")

    training_feature, training_selection = load_data(feature_file)
    validation_feature, validation_selection = load_data(validation_file)

    n_in = training_feature.get_value(borrow=True).shape[2]
    n_out = training_selection.get_value(borrow=True).shape[2]

    # model = SelectionLayer(x, n_in, n_out, class_prob)
    model = SelectionLayer(x, n_in, n_out, None)
    cost = model.negative_log_likelihood(y) + L1_reg * abs(model.W).sum()
    # lik = model.selection_weight

    idx = T.lscalar()  # index to a [mini]batch

    updates = RMSprop(cost, [model.W, model.b], learning_rate);

    train_model = theano.function(
        inputs=[idx],
        outputs=cost,
        updates=updates,
        givens={
            x: training_feature[idx * batch_size : (idx + 1) * batch_size],
            y: training_selection[idx * batch_size : (idx + 1) * batch_size],
        }
    )

    validate_model = theano.function(
        inputs=[idx],
        outputs=cost,
        givens={
            x: validation_feature[idx * batch_size: (idx + 1) * batch_size],
            y: validation_selection[idx * batch_size: (idx + 1) * batch_size]
        }
    )


    ################## train the model #####################
    n_train_batches = training_feature.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = validation_feature.get_value(borrow=True).shape[0] // batch_size

    best_params = None
    best_validation_loss = numpy.inf

    for epoch in xrange(0, nb_epoch):
        train_loss = [train_model(i) for i in range(n_train_batches)];

        validation_losses = [validate_model(i)
                            for i in range(n_valid_batches)]

        this_train_loss = numpy.mean(train_loss)
        this_validation_loss = numpy.mean(validation_losses)
        params = [model.W.get_value(), model.b.get_value()]

        print("epoch = " + str(epoch) + "; training error = " + str(this_train_loss)
                + "; validation error = " + str(this_validation_loss) + "\n")

        if this_validation_loss < best_validation_loss:
            best_params = copy.deepcopy(params)
            best_validation_loss = this_validation_loss

            print ('\n'.join('\t'.join(str(cell) for cell in row)
                            for row in best_params[0]))
            print ('\t'.join(str(cell) for cell in best_params[1]))


    print ('\n'.join('\t'.join(str(cell) for cell in row)
                     for row in best_params[0]))
    print ('\t'.join(str(cell) for cell in best_params[1]))

# load data
def load_data(feature_file):
    feature_data, selection_data = pickle.load(open(feature_file, "rb"))

    shared_feature_data = theano.shared(numpy.asarray(feature_data,
                                            dtype=theano.config.floatX),
                                borrow=True)

    shared_selection_data = theano.shared(numpy.asarray(selection_data,
                                            dtype=theano.config.floatX),
                                borrow=True)

    return shared_feature_data, shared_selection_data

def predict_fitness(para_file, variant_file):
    # read parameter file
    para = genfromtxt(para_file)
    weight = para[0:-1,]
    bias = para[-1, ]

    # build model
    x = T.tensor3("x")
    y = T.tensor3("y")

    n_in = weight.shape[0]
    n_out = weight.shape[1] + 1

    model = SelectionLayer(x, n_in, n_out, None)
    pred_matrix = model.prediction()

    # set parameters
    model.W.set_value(weight)
    model.b.set_value(bias)

    predict_model = theano.function(
        inputs=[x],
        outputs=pred_matrix
    )

    # read data
    with open(variant_file, "r") as f:
        while True:
            next_n_lines = list(islice(f, 10000))
            next_n_lines = [x.rstrip().split("\t") for x in next_n_lines]

            if not next_n_lines:
                break

            feature = [x[5:len(x)] for x in next_n_lines]
            feature = numpy.array(feature, dtype=numpy.float64)
            y = predict_model([feature])
            for i in xrange(y.shape[0]):
                prob = [str(x) for x in y[i]]
                print("\t".join(next_n_lines[i][0:5] + prob))



if __name__ == "__main__":
    if sys.argv[1] == "learn":
        print("\t".join(sys.argv))
        learning_rate = float(sys.argv[4])
        penalty = float(sys.argv[5])
        sgd_optimize_single_layer(learning_rate, sys.argv[2], sys.argv[3], penalty, 100, 100)
    elif sys.argv[1] == "pred":
        predict_fitness(sys.argv[2], sys.argv[3])
