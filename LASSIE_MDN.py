"""
The main program for the DeepINSIGHT model
"""
from __future__ import print_function

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,floatX=float32"
# os.environ["OMP_NUM_THREADS"] = 1

# from itertools import islice
import six.moves.cPickle as pickle
import argparse
import h5py
import numpy
import sys
import copy

import theano
import theano.tensor as T
# from theano.tensor.shared_randomstreams import RandomStreams

# from selection_layer import SelectionLayer, RMSprop, AdaGrad
from selection_layer import RMSprop, AdaGrad
from selection_layer import stochastic_gradient_descent
from hidden_layer import DeepNetwork, predict_fitness


def load_h5_data(feature_file):
    with h5py.File(feature_file, "r") as hf:
        feature_data = numpy.array(hf.get("feature"),
                                   dtype=theano.config.floatX)
        selection_data = numpy.array(hf.get("likelihood"),
                                     dtype=theano.config.floatX)

        shared_feature_data = theano.shared(feature_data,
                                            borrow=True)

        shared_selection_data = theano.shared(selection_data,
                                              borrow=True)

    return shared_feature_data, shared_selection_data


def sgd_optimize(args):
    # read files
    training_feature, training_selection = load_h5_data(args.input_file[0])
    validation_feature, validation_selection = load_h5_data(args.input_file[1])

    x = T.tensor3("x")
    y = T.tensor3("y")

    # parameters for the optimization
    learning_rate = args.learning_rate
    n_hidden = args.num
    L2_reg = args.penalty_L2
    batch_size = args.batch
    nb_epoch = args.epoch
    method = "RMSprop"
    class_prob = None
    activation = theano.tensor.nnet.relu
    autocorrelation = 0.99

    n_in = training_feature.get_value(borrow=True).shape[2]
    n_out = training_selection.get_value(borrow=True).shape[2]

    rng = numpy.random.RandomState()

    # the model for training
    model = DeepNetwork(rng, x, n_in, n_hidden, n_out,
                        class_prob, activation, "training")

    # the model for validation
    model_test = DeepNetwork(rng, x, n_in, n_hidden, n_out,
                             class_prob, activation, "prediction")

    # the cost function for training
    cost = (
        model.negative_log_likelihood(y) + L2_reg * model.L2
    )

    # the cost function for validation
    # note the penalization term should not be used here
    cost_test = (
        model_test.negative_log_likelihood(y)
    )

    idx = T.lscalar()  # index to a [mini]batch

    if method == "RMSprop":
        updates = RMSprop(cost, model.params, learning_rate, autocorrelation)
    elif method == "AdaGrad":
        updates = AdaGrad(cost, model.params, learning_rate)
    elif method == "SGD":
        updates = stochastic_gradient_descent(cost, model.params,
                                              learning_rate)
    else:
        raise Exception("Unimplemented optimization method: " + method + "\n")

    train_model = theano.function(
        inputs=[idx],
        outputs=cost,
        updates=updates,
        givens={
            x: training_feature[idx * batch_size: (idx + 1) * batch_size],
            y: training_selection[idx * batch_size: (idx + 1) * batch_size],
        }
    )

    # the validation model. Note that the cost is different from the training
    # one because of the usage of dropout
    validate_model = theano.function(
        inputs=[idx],
        outputs=cost_test,
        givens={
            x: validation_feature[idx * batch_size: (idx + 1) * batch_size],
            y: validation_selection[idx * batch_size: (idx + 1) * batch_size]
        }
    )

    # train the model
    n_train_batches = training_feature.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = validation_feature.get_value(borrow=True).shape[0] // batch_size

    best_params = None
    best_validation_loss = numpy.inf

    patience = 10

    for epoch in xrange(0, nb_epoch):
        train_loss = [train_model(i) for i in range(n_train_batches)]

        # set parameters for validation
        params = [tmp.get_value() for tmp in model.params]
        for a, b in zip(model_test.params, params):
            a.set_value(b)

        validation_losses = [validate_model(i) for i in range(n_valid_batches)]

        this_train_loss = numpy.mean(train_loss)
        this_validation_loss = numpy.mean(validation_losses)

        print("epoch = " + str(epoch) + "; training error = "
              + str(this_train_loss)
              + "; validation error = "
              + str(this_validation_loss))

        if this_validation_loss < best_validation_loss:
            best_params = copy.deepcopy(params)
            best_validation_loss = this_validation_loss
            patience = 10
        else:
            patience -= 1

        if patience == 0:
            break

    print("Best validation error = " + str(best_validation_loss))
    best_params = [n_in, n_out, n_hidden, best_params]
    pickle.dump(best_params, open(args.output_file, "wb"))


def evaluate_model(args):
    # read files
    validation_feature, validation_selection = load_h5_data(args.input_file)

    x = T.tensor3("x")
    y = T.tensor3("y")

    batch_size = 100

    n_in, n_out, n_hidden, params = pickle.load(open(args.parameter, "rb"))
    class_prob = None
    activation = theano.tensor.nnet.relu

    rng = numpy.random.RandomState(None)
    model_test = DeepNetwork(rng, x, n_in, n_hidden, n_out, class_prob,
                             activation, "prediction")

    # set parameters
    for a, b in zip(model_test.params, params):
        a.set_value(b)

    # the cost function for validation
    # note the penalization term should not be used here
    cost_test = (
        model_test.negative_log_likelihood(y)
    )

    idx = T.lscalar()  # index to a [mini]batch

    # the validation model. Note that the cost is different from the training
    # one because of the usage of dropout
    validate_model = theano.function(
        inputs=[idx],
        outputs=cost_test,
        givens={
            x: validation_feature[idx * batch_size: (idx + 1) * batch_size],
            y: validation_selection[idx * batch_size: (idx + 1) * batch_size]
        }
    )

    # evaluate model
    n_valid_batches = validation_feature.get_value(borrow=True).shape[0] // batch_size
    validation_losses = [validate_model(i) for i in range(n_valid_batches)]

    test_loss = numpy.mean(validation_losses)

    print("Test error = " + str(test_loss))


def open_bed_file(infile, skip_col):
    with open(infile) as f:
        count = 0
        ncols = 0
        for line in f:
            count += 1
            if ncols == 0:
                token = line.split("\t")
                ncols = len(token)
            if count % 1000000 == 0:
                print("read line " + str(count))

    data = numpy.zeros((count, ncols - skip_col), dtype=numpy.float32)

    with open(infile) as f:
        count = 0
        for line in f:
            token = line.split("\t")
            data[count, :] = [float(x) for x in token[skip_col:]]
            count += 1
            if count % 1000000 == 0:
                print("read data for line " + str(count))

    dim = ncols - skip_col
    data = numpy.reshape(data, (-1, 3, dim))
    return data


def convert_data(args):
    # read feature files
    feature = open_bed_file(args.feature, 7)

    # read likelihood files
    likelihood = open_bed_file(args.lik, 5)

    print("data reading finished")
    order = numpy.random.permutation(feature.shape[0])
    print("permutation finished")
    feature = feature[order, :, :]
    print("data1 finished")
    likelihood = likelihood[order, :, :]
    print("data2 finished")
    with h5py.File(args.output_file, 'w') as hf:
        hf.create_dataset('feature', data=feature)
        hf.create_dataset('likelihood', data=likelihood)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='name')

    ################### the parser for training ####################
    parser_learn = subparsers.add_parser("learn")

    parser_learn.add_argument("-i", dest="input_file", nargs=2, type=str, required=True,
                              help="input files for training and validation (HDF5 format)")

    parser_learn.add_argument("-o", dest="output_file", type=str, required=True,
                              help="output file for estimated parameters")

    parser_learn.add_argument("-r", dest="learning_rate", default=0.0001,
                              type=float, help="learning rate of the SGD algorithm")

    parser_learn.add_argument("-n", dest="num", nargs="+", type=int,
                              default=[0], help="numbers of hidden units")

    parser_learn.add_argument("-p", dest="penalty_L2", type=float, default=0.,
                              help="L2 penalty (weight decay)")

    parser_learn.add_argument("-s", dest="seed", type=int,
                              help="random number seed")

    parser_learn.add_argument("-b", dest="batch", type=int,
                              help="mini-batch size", default=100)

    parser_learn.add_argument("-e", dest="epoch", type=int,
                              help="number of epochs", default=100)

    ################### the parser for prediction ####################
    parser_predict = subparsers.add_parser('predict')

    parser_predict.add_argument("-p", dest="parameter", type=str, required=True,
                                help="parameter file")

    parser_predict.add_argument("-f", dest="feature", type=str, required=True,
                                help="feature file for prediction")

    ################### the parser for evaluation ####################
    parser_eval = subparsers.add_parser('eval')

    parser_eval.add_argument("-p", dest="parameter", type=str, required=True,
                             help="parameter file")

    parser_eval.add_argument("-i", dest="input_file", type=str, required=True,
                             help="input file for evaluation (HDF5 format)")

    ################### the parser for data conversion ####################
    parser_convert = subparsers.add_parser('convert')

    parser_convert.add_argument("-f", dest="feature", type=str, required=True,
                                help="feature file for training or validation")

    parser_convert.add_argument("-l", dest="lik", type=str, required=True,
                                help="likelihood file for training or validation")

    parser_convert.add_argument("-o", dest="output_file", type=str, required=True,
                                help="output file (HDF5 format)")

    parser_convert.add_argument("-s", dest="seed", type=int,
                                help="random number seed")

    args = parser.parse_args()

    if args.name == "learn":
        print(" ".join(["CMD:"] + sys.argv))
        numpy.random.seed(args.seed)
        sgd_optimize(args)
    elif args.name == "convert":
        print(" ".join(["CMD:"] + sys.argv))
        numpy.random.seed(args.seed)
        convert_data(args)
    elif args.name == "predict":
        predict_fitness(args.parameter, args.feature)
    elif args.name == "eval":
        evaluate_model(args)
