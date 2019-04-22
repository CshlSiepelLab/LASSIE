"""
The hidden layer used in the DeepINSIGHT model
"""

from __future__ import print_function

from itertools import islice

import six.moves.cPickle as pickle
import numpy
import sys
import copy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from selection_layer import SelectionLayer, RMSprop, AdaGrad, load_data
from selection_layer import stochastic_gradient_descent


# The hidden layer based on Theano tutorial
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=theano.tensor.nnet.relu):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # Initial weights. If the activation functions are either tanh
        # or logistic, use the method in deep learning tutorial.
        # If ReLU is used, we use the initialization method by Kaiming He
        # which is a Gaussian distribution with mean = 0 and SD =
        # sqrt(2/(n_in+n_out)).
        # Reference:
        # http://arxiv.org/pdf/1502.01852v1.pdf
        # http://stackoverflow.com/questions/33229222/how-to-initialize-weights-when-using-relu-activation-function
        if W is None:
            if activation == theano.tensor.nnet.relu:
                W_values = numpy.asarray(
                    rng.randn(
                        n_in, n_out
                    ),
                    dtype=theano.config.floatX
                )

                W_values *= numpy.sqrt(2. / (n_in + n_out))
            else:
                W_values = numpy.asarray(
                    rng.uniform(
                        low=-numpy.sqrt(6. / (n_in + n_out)),
                        high=numpy.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
                )

                if activation == theano.tensor.nnet.sigmoid:
                    W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

# a deep network using the PRF likelihood in pop. gen.
class DeepNetwork(object):
    # def __init__(self, rng, input, n_in, n_hidden_list, n_out, activation=T.nnet.relu):
    def __init__(self, rng, input, n_in, n_hidden_list, n_out, class_prob=None,
                 activation=T.tanh, dropout=""):
        self.hiddenLayer = []

        if len(n_hidden_list) == 1 and n_hidden_list[0] == 0:
            # The selection layer
            self.selectionLayer = SelectionLayer(input=input,
                                                n_in=n_in,
                                                n_out=n_out,
                                                class_prob=class_prob
                                                )
        else:
            # construct the first hidden layer
            layer = HiddenLayer(
                rng=rng,
                input=input,
                n_in=n_in,
                n_out=n_hidden_list[0],
                activation=activation
            )

            srng = RandomStreams()
            if dropout == "training":
                layer.output = T.switch(srng.binomial(size=(n_hidden_list[0],), p=0.5), layer.output, 0)
            elif dropout == "prediction":
                layer.output = 0.5 * layer.output
            elif dropout != "":
                raise Exception("Unkown dropout option!\n")

            self.hiddenLayer.append(layer)

            # construct all rest hidden layers
            for in_size, out_size in zip(n_hidden_list[:-1], n_hidden_list[1:]):
                layer = HiddenLayer(
                    rng=rng,
                    input=self.hiddenLayer[-1].output,
                    n_in=in_size,
                    n_out=out_size,
                    activation=activation
                )
                if dropout == "training":
                    layer.output = T.switch(srng.binomial(size=(out_size,), p=0.5), layer.output, 0)
                elif dropout == "prediction":
                    layer.output = 0.5 * layer.output
                elif dropout != "":
                    raise Exception("Unkown dropout option!\n")

                self.hiddenLayer.append(layer)

            # The selection layer
            self.selectionLayer = SelectionLayer(input=self.hiddenLayer[-1].output,
                                                n_in=n_hidden_list[-1],
                                                n_out=n_out,
                                                class_prob=class_prob
                                                )
        # L1 reg
        self.L1 = abs(self.selectionLayer.W).sum()
        for layer in self.hiddenLayer:
            self.L1 = self.L1 + abs(layer.W).sum()

        # L2 reg
        self.L2 = (self.selectionLayer.W ** 2).sum()
        for layer in self.hiddenLayer:
            self.L2 = self.L2 + (layer.W ** 2).sum()

        # the objective function
        self.negative_log_likelihood = (
            self.selectionLayer.negative_log_likelihood
        )

        # All parameters
        self.params = self.selectionLayer.params
        for layer in self.hiddenLayer:
            self.params = layer.params + self.params

        # keep track of model input
        self.input = input

    def prediction(self):
        return self.selectionLayer.selection_weight


# single layer MLP using the PRF likelihood in pop. gen.
class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out, activation=T.tanh):
        """ Initialize the parameters for MLP
        """

        # The hidden layer
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=activation
        )

        # The selection layer
        self.selectionLayer = SelectionLayer(input=self.hiddenLayer.output,
                                             n_in=n_hidden,
                                             n_out=n_out,
                                             class_prob=None
                                             )


        # L1 reg
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.selectionLayer.W).sum()
        )

        # L2 reg
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.selectionLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.selectionLayer.negative_log_likelihood
        )

        # All parameters
        self.params = self.hiddenLayer.params + self.selectionLayer.params

        # keep track of model input
        self.input = input

    def prediction(self):
        return self.selectionLayer.selection_weight


def sgd_optimize_deep_network(learning_rate, feature_file, validation_file, n_hidden,
                             L2_reg=0.0, batch_size=100, nb_epoch=100, method="RMSprop",
                             class_prob=None, activation=theano.tensor.nnet.relu):
    x = T.tensor3("x")
    y = T.tensor3("y")

    training_feature, training_selection = load_data(feature_file)
    validation_feature, validation_selection = load_data(validation_file)

    n_in = training_feature.get_value(borrow=True).shape[2]
    n_out = training_selection.get_value(borrow=True).shape[2]

    # rng = numpy.random.RandomState(1234)
    rng = numpy.random.RandomState()

    # the model for training
    model = DeepNetwork(rng, x, n_in, n_hidden, n_out, class_prob, activation, "training")

    # the model for validation
    model_test = DeepNetwork(rng, x, n_in, n_hidden, n_out, class_prob, activation, "prediction")

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
        updates = RMSprop(cost, model.params, learning_rate, 0.9)
    elif method == "AdaGrad":
        updates = AdaGrad(cost, model.params, learning_rate)
    elif method == "SGD":
        updates = stochastic_gradient_descent(cost, model.params, learning_rate)
    else:
        raise Exception("Unimplemented optimization method: " + method + "\n")

    train_model = theano.function(
        inputs=[idx],
        outputs=cost,
        updates=updates,
        givens={
            x: training_feature[idx * batch_size : (idx + 1) * batch_size],
            y: training_selection[idx * batch_size : (idx + 1) * batch_size],
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


    ################## train the model #####################
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

        validation_losses = [validate_model(i)
                            for i in range(n_valid_batches)]

        this_train_loss = numpy.mean(train_loss)
        this_validation_loss = numpy.mean(validation_losses)

        print("epoch = " + str(epoch) + "; training error = " + str(this_train_loss)
                + "; validation error = " + str(this_validation_loss))

        if this_validation_loss < best_validation_loss:
            best_params = copy.deepcopy(params)
            best_validation_loss = this_validation_loss
            patience = 10
        else:
            patience -= 1

        if patience == 0:
            break

    print("Best validation error = " + str(best_validation_loss))
    return [n_in, n_out, n_hidden, best_params]

def eval_feature_importance(para_file, test_file, batch_size=100,
                            activation=theano.tensor.nnet.relu
                            ):
    test_feature, test_selection = load_data(test_file)
    x = T.tensor3("x")
    y = T.tensor3("y")

    n_in, n_out, n_hidden, params = pickle.load(open(para_file, "rb"))

    # model = SelectionLayer(x, n_in, n_out, None)
    rng = numpy.random.RandomState(None)
    # model = MLP(rng, x, n_in, n_hidden, n_out)
    model = DeepNetwork(rng, x, n_in, n_hidden, n_out, None, activation, "prediction")

    cost_test = (
        model.negative_log_likelihood(y)
    )

    # set parameters
    for a, b in zip(model.params, params):
        a.set_value(b)

    idx = T.lscalar()  # index to a [mini]batch

    test_model = theano.function(
        inputs=[idx],
        outputs=cost_test,
        givens={
            x: test_feature[idx * batch_size: (idx + 1) * batch_size],
            y: test_selection[idx * batch_size: (idx + 1) * batch_size]
        }
    )

    n_test_batches = test_feature.get_value(borrow=True).shape[0] // batch_size
    test_losses = [test_model(i) for i in range(n_test_batches)]
    ori_loss = numpy.mean(test_losses)
    print("# original loss = " + str(ori_loss))

    ori_data = numpy.copy(test_feature.get_value(borrow=True))
    for i in xrange(ori_data.shape[2]):
        # print(i)
        tmp_data = numpy.copy(ori_data)
        tmp_data[:, :, i] = numpy.mean(tmp_data[:, :, i])
        test_feature.set_value(tmp_data)
        test_losses = [test_model(i) for i in range(n_test_batches)]
        new_loss = numpy.mean(test_losses)
        print(new_loss - ori_loss)

def predict_fitness(para_file, variant_file, activation=theano.tensor.nnet.relu):
    # build model
    x = T.tensor3("x")
    y = T.tensor3("y")

    n_in, n_out, n_hidden, params = pickle.load(open(para_file, "rb"))

    rng = numpy.random.RandomState(None)
    model = DeepNetwork(rng, x, n_in, n_hidden, n_out, None, activation, "prediction")
    pred_matrix = model.prediction()

    # set parameters
    for a, b in zip(model.params, params):
        a.set_value(b)

    predict_model = theano.function(
        inputs=[x],
        outputs=pred_matrix
    )

    # read data
    with open(variant_file, "r") as f:
        while True:
            next_n_lines = list(islice(f, 10000))
            next_n_lines = [tmp.rstrip().split("\t") for tmp in next_n_lines]

            if not next_n_lines:
                break

            feature = [tmp[5:len(tmp)] for tmp in next_n_lines]
            feature = numpy.array(feature, dtype=theano.config.floatX)
            y = predict_model([feature])
            for i in xrange(y.shape[0]):
                prob = [str(tmp) for tmp in y[i]]
                print("\t".join(next_n_lines[i][0:5] + prob))


if __name__ == "__main__":
    if sys.argv[1] == "learn":
        print(" ".join(["CMD: "] + sys.argv))
        learning_rate = float(sys.argv[4])
        n_hidden = [int(x) for x in sys.argv[5].split(",")]
        penalty = float(sys.argv[6])
        outfile = sys.argv[7]

        method = sys.argv[8]
        class_prob = [float(x) for x in sys.argv[9].split(",")]

        best_params = sgd_optimize_deep_network(learning_rate, sys.argv[2], sys.argv[3],
                                                n_hidden, penalty, 100, 100, method, class_prob)
        pickle.dump(best_params, open(outfile, "wb"))
    elif sys.argv[1] == "pred":
        predict_fitness(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "import":
        eval_feature_importance(sys.argv[2], sys.argv[3])
