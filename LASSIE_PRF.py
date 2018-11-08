"""
The python implementation of Poisson random field model
"""
from __future__ import print_function, division

import sys
import time
import numpy
import scipy
import argparse
from scipy.stats import binom, hypergeom
from scipy.linalg import solve_banded
from collections import OrderedDict
from scipy.optimize import minimize


class PoissonRandomFieldModel(object):
    """
    The class to calculate theoretical SFS based on Evans 2007
    """
    def __init__(self, K):
        """
        : K: the number of bins
        """
        self.K = K
        self.varepsilon = 1. / K

        self.diag = numpy.zeros(K + 1)
        self.U = numpy.zeros(K)
        self.V = numpy.zeros(K)

        self.diag2 = numpy.zeros(K + 1)
        self.U2 = numpy.zeros(K)
        self.V2 = numpy.zeros(K)

        self.gene_frequency = numpy.zeros(K + 1)

    def solve(self, pop_size, duration, gamma):
        """
        sovle the forward equation with the boundary condition in Evans 2007

        : pop_size: population sizes (python list)
        : duration: durations (python list)
        : gamma: scaled selection coefficients (float)
        """
        theta = 1.
        if len(pop_size) != len(duration):
            raise Exception("Unequal size of pop_size and duration!\n")

        # freq = numpy.zeros(self.K + 1)
        tmp_freq = numpy.zeros(self.K + 1)

        # initialize the density using stationary distribution
        if gamma == 0.:
            x = self.varepsilon * numpy.asarray(range(0, self.K + 1), dtype=float)
            self.gene_frequency = theta * (1. - x)
        else:
            x = self.varepsilon * numpy.asarray(range(0, self.K + 1), dtype=float)

            self.gene_frequency = (theta * numpy.exp(2. * gamma)
                                   * (1. - numpy.exp(-2. * gamma * (1. - x)))
                                   / (numpy.exp(2. * gamma) - 1.))
        prev_p1 = -1.
        prev_p2 = -1.
        prev_tau = -1.

        tmp_tridiag = numpy.zeros((3, self.K + 1))

        for i in xrange(0, len(pop_size)):
            if i == 0:
                p1 = 1.
            else:
                p1 = pop_size[i - 1]
            p2 = pop_size[i]
            tau = duration[i]

            # update matrices if required
            if p1 != prev_p1 or p2 != prev_p2 or tau != prev_tau:
                self.compute_crank_matrix(p1, p2, tau, gamma, theta)
                tmp_tridiag[1, :] = self.diag
                tmp_tridiag[0, 1:] = self.U
                tmp_tridiag[2, :-1] = self.V

            # solve the tridiagonal system here
            tmp_freq[1:-1] = (self.gene_frequency[1:-1] * self.diag2[1:-1]
                              + self.gene_frequency[2:] * self.U2[1:]
                              + self.gene_frequency[0:-2] * self.V2[0:-1])

            tmp_freq[0] = (self.gene_frequency[0] * self.diag2[0]
                           + self.gene_frequency[1] * self.U2[0])

            tmp_freq[-1] = (self.gene_frequency[-1] * self.diag2[-1]
                            + self.gene_frequency[-2] * self.V2[-1])

            self.gene_frequency = solve_banded((1, 1), tmp_tridiag, tmp_freq, check_finite=False)
            prev_p1 = p1
            prev_p2 = p2
            prev_tau = tau

        # note the frequency need to be transformed before use
        x = self.varepsilon * numpy.asarray(range(1, self.K), dtype=float)
        self.gene_frequency[1:-1] = self.gene_frequency[1:-1] / (x * (1. - x))

    def diffusion_selection(self, n, gamma):
        """
        The selection term in diffusion approximation
        """
        return (-1. * (gamma) * (self.varepsilon * (n)) * (1. - self.varepsilon * (n)))

    def diffusion_drift(self, n, pop):
        """
        The drif term in diffusion approximation
        """
        return ((self.varepsilon * (n)) * (1. - self.varepsilon * (n)) / (2. * (pop)))

    def compute_crank_matrix(self, pop_size, pop_size_next, tau, gamma, theta):
        """
        construct the matrices in the Crank-Nicolson algorithm

        : pop_size: population size at time t
        : pop_size_next: population size at time t+1
        : tau: duration fromt time t to t+1
        : gamma: scaled selection coefficient
        : theta: 4Ne\mu in the ancestral population
        """
        delta = 0.5 * tau / (self.varepsilon * self.varepsilon)
        rho = 0.25 * tau / self.varepsilon

        # main diagonal
        for i in xrange(self.K + 1):
            value = 1. + 2. * delta * self.diffusion_drift(i, pop_size_next)
            value2 = 1. - 2. * delta * self.diffusion_drift(i, pop_size)
            if i == 0:
                # Note that it is essential change boundary condition here
                # if population size changes
                value = 1.
                value2 = pop_size_next / pop_size
            elif i == self.K:
                value = value2 = 1.

            self.diag[i] = value
            self.diag2[i] = value2

        # leading upper diagonal
        for i in xrange(self.K):
            value = (- delta * self.diffusion_drift(i, pop_size_next)
                     - rho * self.diffusion_selection(i, gamma))

            value2 = (delta * self.diffusion_drift(i, pop_size)
                      + rho * self.diffusion_selection(i, gamma))

            if i == 0:
                value = value2 = 0.

            self.U[i] = value
            self.U2[i] = value2

        # leading lower diagonal
        for i in xrange(self.K):
            value = (- delta * self.diffusion_drift(i + 1,  pop_size_next)
                     + rho * self.diffusion_selection(i + 1, gamma))
            value2 = (delta * self.diffusion_drift(i + 1,  pop_size)
                      - rho * self.diffusion_selection(i + 1, gamma))

            if i == self.K - 1:
                value = value2 = 0.

            self.V[i] = value
            self.V2[i] = value2

    def unfolded_sfs(self, theta, sample_size):
        """
        calculate unfolded site frequency spectrum
        """
        w = numpy.ones(self.K - 1, dtype=float)
        w[0] = w[-1] = 55. / 24.
        w[1] = w[-2] = -1. / 6.
        w[2] = w[-3] = 11. / 8.
        x = self.varepsilon * numpy.asarray(range(1, self.K), dtype=float)
        sfs = [0.]
        for i in xrange(1, sample_size):
            prob = self.varepsilon * theta * numpy.sum(w * binom.pmf(i, sample_size, x) * self.gene_frequency[1:-1])
            sfs.append(prob)

        return numpy.asarray(sfs, dtype=float)


class PiecewiseConstantModel(object):
    """
    The piecewise constant demographic model based on Poisson random field.
    """
    def __init__(self, ref_count, alt_count, ref_prior, alt_prior, pop_size,
                 duration, relative_gamma, gamma_weight, theta, polymorphism_only=False,
                 tau=1.e-4, nbin=1000):
        """
        : ref_count: count of reference alleles (numpy array, int)
        : alt_count: count of alternative allels (numpy array, int)
        : ref_prior: prior probability of reference alleles (numpy array, float)
        : alt_prior: prior probability of alternative alleles (numpy array, float)
        : pop_size: population sizes in epochs (python list)
        : duration: durations in epochs (python list)
        : relative_gamma: scaled selection coefficient (transformed tobox constraint)
        : gamma_weight: weights of gamma (transformed to box constraint)
        : theta: 4Ne\mu in the ancestral population
        : polymorphism_only: if True, only use polymorphism for demographic inference
        : tau: bin size for time
        : nbin: number of bins for gene frequency
        """
        self.ref_count = ref_count
        self.alt_count = alt_count
        self.ref_prior = ref_prior
        self.alt_prior = alt_prior
        self.polymorphism_only = polymorphism_only
        self.tau = tau
        self.nsel = len(relative_gamma) + 1

        # initialize parameters
        self.parameter = OrderedDict()

        for i, value in enumerate(pop_size):
            name = "pop_size_" + str(i)
            self.parameter.update({name: [value, 1.e-2, 100.]})

        for i, value in enumerate(duration):
            name = "duration_" + str(i)
            self.parameter.update({name: [value, 1.e-3, 2.]})

        for i, value in enumerate(relative_gamma):
            name = "relative_gamma_" + str(i)
            self.parameter.update({name: [value, -100., -1.e-3]})

        for i, value in enumerate(gamma_weight):
            name = "gamma_weight_" + str(i)
            self.parameter.update({name: [value, 1.e-3, 1 -1.e-3]})

        self.parameter.update({"theta": [theta, 1.e-10, 0.01]})

        # calculate the maximum sample size
        self.max_sample_size = numpy.amax(ref_count + alt_count)

        self.sfs_matrix = numpy.zeros((self.max_sample_size + 1, self.max_sample_size))

        # create PDE model
        self.model = PoissonRandomFieldModel(nbin)

        self.fire_parameter_changed()

    def fire_parameter_changed(self):
        """
        function to trigger computation of likelihood
        """
        # construct the history
        size, time = self.build_demographic_history()

        # calculate all SFS for different sample size
        self.update_sfs(size, time)

        # calculate the log likelihood
        total = self.ref_count + self.alt_count

        mono_site = numpy.logical_or(self.ref_count == 0, self.alt_count == 0)
        poly_site = numpy.logical_and(self.ref_count > 0, self.alt_count > 0)

        # mask monomorphic sites when only polymorphism data are used
        if self.polymorphism_only:
            mono_site.fill(False)

        p1 = (self.sfs_matrix[total[poly_site], self.alt_count[poly_site]]
              * self.ref_prior[poly_site])

        p2 = (self.sfs_matrix[total[poly_site], self.ref_count[poly_site]]
              * self.alt_prior[poly_site])
        p3 = self.sfs_matrix[total[mono_site], 0]
        site_likelihood = numpy.concatenate((p1 + p2, p3))
        site_likelihood[site_likelihood < 1.e-18] = 1.e-18
        self.log_likelihood = numpy.sum(numpy.log(site_likelihood))

    def update_sfs(self, size, time):
        """
        function to update the SFS matrix
        """
        # reset value to zeros
        self.sfs_matrix.fill(0.)

        # calculate the SFS of the maximum sample size
        self.sfs_matrix[-1, :] = self.calculate_one_sfs(size, time, self.max_sample_size)

        tmp_matrix = numpy.zeros((self.max_sample_size, self.max_sample_size), dtype=float)

        lower_sample_size = 2
        # loop over sample size
        for i in xrange(lower_sample_size, self.max_sample_size):
            tmp_matrix.fill(0.)
            tmp_vector = numpy.arange(1, i)
            # loop over frequency in the maximum sample
            for j in xrange(1, self.max_sample_size):
                x = hypergeom.pmf(tmp_vector, self.max_sample_size, j, i)
                tmp_matrix[1:i, j] = x

            x = numpy.sum(tmp_matrix * self.sfs_matrix[-1, :], 1)
            self.sfs_matrix[i, :] = x

        # for the sample sizes which are not covered by the computation above
        # set the probability to 0, important for making sure the correctness
        # of the mixture density network
        self.sfs_matrix[0:lower_sample_size, :].fill(0.)

        # transform final probability based on data type
        row_sum =  numpy.sum(self.sfs_matrix[lower_sample_size:, :], 1)
        if self.polymorphism_only:
            self.sfs_matrix[lower_sample_size:, :] /= row_sum[:, numpy.newaxis]
        else:
            self.sfs_matrix[lower_sample_size:, 0] = 1. - row_sum

    def calculate_one_sfs(self, size, time, n_sample):
        tmp_matrix = numpy.zeros((self.nsel, n_sample))
        gamma = [0]
        weight = []

        prob = 1.
        for x in self.parameter:
            if x[0:14] == "relative_gamma":
                gamma.append(gamma[-1] + self.parameter[x][0])

            if x[0:12] == "gamma_weight":
                weight.append(prob * self.parameter[x][0])
                prob = prob * (1. - self.parameter[x][0])

        weight.insert(0, prob)

        theta = self.parameter["theta"][0]

        for i, value in enumerate(gamma):
            w = weight[i]
            self.model.solve(size, time, value)
            tmp_matrix[i, :] = w * self.model.unfolded_sfs(theta, n_sample)

        return(numpy.sum(tmp_matrix, 0))

    def build_demographic_history(self):
        pop_size = []
        duration = []
        for x in self.parameter:
            if x[0:8] == "duration":
                duration.append(self.parameter[x][0])

            if x[0:8] == "pop_size":
                pop_size.append(self.parameter[x][0])

        size = []
        time = []
        for n, t in zip(pop_size, duration):
            for x in numpy.arange(0., t, self.tau):
                if x + self.tau <= t:
                    d = self.tau
                else:
                    d = t - x

                size.append(n)
                time.append(d)

        return size, time

    def optimize(self, mode, esp=1.e-6):
        """
        The function to optimize log likelihood conditional on a list
        of parameters.

        para_list: a list of parameter names for optimization
        """
        self.para_for_optimize = []
        if mode == "demo":
            for name in self.parameter:
                if name[0:8] == "duration":
                    self.para_for_optimize.append(name)

                if name[0:8] == "pop_size":
                    self.para_for_optimize.append(name)

            # self.polymorphism_only = True
        elif mode == "theta":
            self.para_for_optimize.append("theta")
            # self.polymorphism_only = False
        elif mode == "sel":
            for name in self.parameter:
                if name[0:14] == "relative_gamma":
                    self.para_for_optimize.append(name)

                if name[0:12] == "gamma_weight":
                    self.para_for_optimize.append(name)
            # self.polymorphism_only = False
        else:
            raise Exception("Unknown mode of optimization: " + model + "!!\n")

        initial_para = []
        boundary = []
        for name in self.para_for_optimize:
            initial_para.append(self.parameter[name][0])
            boundary.append((self.parameter[name][1], self.parameter[name][2]))

        def callback(x):
            print("optimizing ...")

        res = minimize(self, x0=numpy.asarray(initial_para), bounds=boundary,
                       jac=None, callback=callback, options={"eps": esp})

        # set parameters to the optimal values
        self(res.x)
        return(res.success)

    def __call__(self, para):
        """
        The functor.

        para: numpy array of intial parameters.
        """
        if len(self.para_for_optimize) != para.size:
            raise Exception("Unequal vector size!!\n")

        for name, value in zip(self.para_for_optimize, para.tolist()):
            self.parameter[name][0] = value

        self.fire_parameter_changed()
        return -self.log_likelihood


def read_data(infile):
    ref_count = []
    alt_count = []
    ref_prior = []
    alt_prior = []
    with open(infile) as f:
        for line in f:
            item = line.split("\t")
            ref_count.append(int(item[5]))
            alt_count.append(int(item[6]))
            ref_prior.append(float(item[7]))
            alt_prior.append(float(item[8]))

    return (numpy.asarray(ref_count), numpy.asarray(alt_count),
            numpy.asarray(ref_prior), numpy.asarray(alt_prior))

def convert_parameter_to_internal(gamma, weight):
    """
    convert parameters to internal representation.
    """
    # convert gamma
    outgamma = []
    cur = 0.
    for x in gamma:
        if x >= 0.:
            print("Error: invalid gamma " + str(x))
            sys.exit(1)

        outgamma.append(x - cur)
        cur = x

    # convert weight
    outweight = []
    cur = 1.
    for x in weight:
        if x >= 1.0 or x <= 0.0:
            print("Error: invalid weight " + str(x))
            sys.exit(1)

        outweight.append(x / cur)
        cur = cur - x

    return outgamma, outweight

def convert_parameter_to_external(internal_gamma, internal_weight):
    """
    convert parameters to external representation.
    """
    # convert gamma
    gamma = []
    cur = 0.
    for x in internal_gamma:
        cur = x + cur
        gamma.append(cur)

    # convert weight
    weight = []
    cur = 1.
    for x in internal_weight:
        weight.append(cur * x)
        cur = cur - cur * x

    return gamma, weight

def run_demo_inference(args):
    if len(args.duration) != len(args.size):
        print("Error: unequal lengths of duration and population size vectors")
        sys.exit(1)

    ref_count, alt_count, ref_prior, alt_prior = read_data(args.infile)

    demo_model = PiecewiseConstantModel(ref_count, alt_count, ref_prior, alt_prior,
                                        args.size, args.duration, [], [], 1., True)
    print("Mode: inference of neutral demographic model")
    status = demo_model.optimize("demo")
    return (demo_model.parameter, demo_model.log_likelihood, status,
            demo_model.sfs_matrix[-1].tolist())

def run_theta_inference(args):
    if len(args.duration) != len(args.size):
        print("Error: unequal lengths of duration and population size vectors")
        sys.exit(1)

    if args.theta is None:
        print("Error: initial theta must be provided in mode \"theta\"")
        sys.exit(1)

    ref_count, alt_count, ref_prior, alt_prior = read_data(args.infile)

    demo_model = PiecewiseConstantModel(ref_count, alt_count, ref_prior, alt_prior,
                                        args.size, args.duration, [], [], args.theta, False)

    print("Mode: inference of theta")
    status = demo_model.optimize("theta")
    return (demo_model.parameter, demo_model.log_likelihood, status,
            demo_model.sfs_matrix[-1].tolist())

def run_sel_inference(args, polymorphism_only=False):
    if len(args.duration) != len(args.size):
        print("Error: unequal lengths of duration and population size vectors")
        sys.exit(1)

    if args.selection is None or args.weight is None:
        print("Error: initial selection parameters must be provided in \"sel\" mode")
        sys.exit(1)

    if len(args.selection) != len(args.weight):
        print("Error: unequal lengths of selection and weight vectors")
        sys.exit(1)

    ref_count, alt_count, ref_prior, alt_prior = read_data(args.infile)

    gamma, weight = convert_parameter_to_internal(args.selection, args.weight)

    demo_model = PiecewiseConstantModel(ref_count, alt_count, ref_prior, alt_prior,
                                        args.size, args.duration, gamma, weight,
                                        args.theta, polymorphism_only)

    print("Mode: inference of selection coefficients")
    status = demo_model.optimize("sel")
    return (demo_model.parameter, demo_model.log_likelihood, status,
            demo_model.sfs_matrix[-1].tolist())

def calculate_class_likelihood(args):
    if len(args.duration) != len(args.size):
        print("Error: unequal lengths of duration and population size vectors")
        sys.exit(1)

    if args.selection is None:
        print("Error: selection parameters must be provided in \"lik\" mode")
        sys.exit(1)

    # obtain the maximum sample size
    max_sample_size = 0
    with open(args.infile) as f:
        for line in f:
            item = line.split("\t")
            ref_count = int(item[5])
            alt_count = int(item[6])
            if max_sample_size < ref_count + alt_count:
                max_sample_size = ref_count + alt_count

    # pseudo data
    ref_count = numpy.asarray([max_sample_size], dtype=int)
    alt_count = numpy.asarray([0], dtype=int)
    ref_prior = numpy.asarray([1.], dtype=float)
    alt_prior = numpy.asarray([0.], dtype=float)

    # create a vector of models
    # for negative selection classes
    model_set = []
    for gamma in args.selection:
        demo_model = PiecewiseConstantModel(ref_count, alt_count, ref_prior, alt_prior,
                                            args.size, args.duration, [gamma], [1.],
                                            args.theta, False)
        model_set.append(demo_model)

    # for neutral class
    demo_model = PiecewiseConstantModel(ref_count, alt_count, ref_prior, alt_prior,
                                        args.size, args.duration, [], [],
                                        args.theta, False)
    model_set.append(demo_model)

    # read data and calculate likelihood
    with open(args.infile) as f:
        for line in f:
            item = line.split("\t")
            ref_count = int(item[5])
            alt_count = int(item[6])
            likelihood = []
            for m in model_set:
                lik = m.sfs_matrix[ref_count + alt_count, alt_count]

                likelihood.append(lik)
            print("\t".join(item[0:5] + [str(x) for x in likelihood]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='name')

    ################### the parser for PRF ####################

    # demographic model
    parser_demo = subparsers.add_parser("demo")

    parser_demo.add_argument("-d", dest="duration", nargs="+", type=float, required=True,
                             help="initial values of durations of epochs")

    parser_demo.add_argument("-n", dest="size", nargs="+", type=float, required=True,
                             help="initial values of population sizes")

    parser_demo.add_argument("-f", dest="infile", required=True, help="input file")

    # estimate of theta
    parser_theta = subparsers.add_parser("theta")

    parser_theta.add_argument("-d", dest="duration", nargs="+", type=float, required=True,
                              help="estimated durations of epochs")

    parser_theta.add_argument("-n", dest="size", nargs="+", type=float, required=True,
                              help="estimated population sizes")

    parser_theta.add_argument("-t", dest="theta", type=float,
                              help="initial value of theta in the ancestral population")

    parser_theta.add_argument("-f", dest="infile", required=True, help="input file")

    # estimate of selection coefficients
    parser_sel = subparsers.add_parser("sel")

    parser_sel.add_argument("-d", dest="duration", nargs="+", type=float, required=True,
                            help="estimated durations of epochs")

    parser_sel.add_argument("-n", dest="size", nargs="+", type=float, required=True,
                            help="estimated population sizes")

    parser_sel.add_argument("-t", dest="theta", type=float,
                            help="estimated theta in the ancestral population")

    parser_sel.add_argument("-g", dest="selection", nargs="+", type=float,
                            help="initial values of selection coefficients")

    parser_sel.add_argument("-w", dest="weight", nargs="+", type=float,
                            help="intitial values of weights of selection categories")

    parser_sel.add_argument("-f", dest="infile", required=True, help="input file")

    # calculate likelihood
    parser_lik = subparsers.add_parser("lik")

    parser_lik.add_argument("-d", dest="duration", nargs="+", type=float, required=True,
                            help="estimated durations of epochs")

    parser_lik.add_argument("-n", dest="size", nargs="+", type=float, required=True,
                            help="estimated population sizes")

    parser_lik.add_argument("-t", dest="theta", type=float,
                            help="estimated theta in the ancestral population")

    parser_lik.add_argument("-g", dest="selection", nargs="+", type=float,
                            help="estimated selection coefficients")

    parser_lik.add_argument("-w", dest="weight", nargs="+", type=float,
                            help="estimated weights of selection categories")

    parser_lik.add_argument("-f", dest="infile", required=True, help="input file")

    args = parser.parse_args()

    if args.name == "demo":
        print("CMD: " + " ".join(sys.argv) + "\n")
        para, lnl, status, sfs = run_demo_inference(args)
    elif args.name == "theta":
        print("CMD: " + " ".join(sys.argv) + "\n")
        para, lnl, status, sfs = run_theta_inference(args)
    elif args.name == "sel":
        print("CMD: " + " ".join(sys.argv) + "\n")
        para, lnl, status, sfs = run_sel_inference(args)
    elif args.name == "sel_no_invariant":
        print("CMD: " + " ".join(sys.argv) + "\n")
        para, lnl, status, sfs = run_sel_inference(args, True)
    elif args.name == "lik":
        calculate_class_likelihood(args)
        sys.exit(0)

    # print result
    size = []
    duration = []
    gamma = []
    weight = []
    for x in para:
        if x[0:8] == "duration":
            duration.append(para[x][0])

        if x[0:8] == "pop_size":
            size.append(para[x][0])

        if x[0:14] == "relative_gamma":
            gamma.append(para[x][0])

        if x[0:12] == "gamma_weight":
            weight.append(para[x][0])

    gamma, weight = convert_parameter_to_external(gamma, weight)

    print("Optimization finished")
    print("Covergence: " + str(status) + "\n\nBest fitted model:")
    print("pop size:  " + " ".join(["%.12f" % x for x in size]))
    print("duration:  " + " ".join(["%.12f" % x for x in duration]))

    if args.name == "sel" or args.name == "sel_no_invariant":
        print("selection: " + " ".join(["%.12f" % x for x in gamma]))
        print("weight:    " + " ".join(["%.12f" % x for x in weight]))

    if (args.name != "demo"):
        print("theta:     " + "%.12f" % para["theta"][0])

    print("\nlog likelihood = " + "%.6f" % lnl)

    print("\nTheoretical SFS:")
    for i, x in enumerate(sfs):
        print("%d\t%.12f" % (i, x))
