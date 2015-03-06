"""
Plot Poisson rate posterior PDFs, and binomial alpha posterior PDFs, as
a demo of the UnivariateBayesianInference class.

Created Feb 27, 2015 by Tom Loredo
"""

from scipy import *

from univariate_bayes import UnivariateBayesianInference


class BinomialInference(UnivariateBayesianInference):
    """
    Bayesian inference for the probability of a Bernoulli outcome, based
    on binomial data.
    """
 
    def __init__(self, n, n_trials, prior=1., na=200, arange=(0.,1.)):
        """
        Define a posterior PDF for the probability of a Bernoulli outcome,
        alpha, based on binomail data.

        Parameters
        ----------

        n : int
            Number of successes

        n_trials : int
            Number of trials (>= n)

        prior : const or function
            Prior PDF for alpha, as a constant for flat prior, or
            a function that can evaluate the PDF on an array
        """
        self.n, self.n_trials = n, n_trials
        self.na = na
        self.alphas = linspace(arange[0], arange[1], na)

        # Pass info to the base class initializer.
        super(BinomialInference, self).__init__(self.alphas, prior, self.lfunc)

    def lfunc(self, alphas):
        """
        Evaluate the Binomial likelihood function on a grid of alphas.
        """
        # Ignore the combinatorial factor (indep. of alpha).
        return (alphas)**self.n * (1.-alphas)**(self.n_trials - self.n)


class PoissonRateInference(UnivariateBayesianInference):
    """
    Bayesian inference for a Poisson rate.
    """
 
    def __init__(self, intvl, n, prior, r_u, r_l=0, nr=200):
        """
        Define a posterior PDF for a Poisson rate.

        Parameters
        ----------
        intvl : float
            Interval for observations

        n : int
            Counts observed

        prior : const or function
            Prior PDF for the rate, as a constant for flat prior, or
            a function that can evaluate the PDF on an array

        r_u : float
            Upper limit on rate for evaluating the PDF
        """
        self.intvl = intvl
        self.n = n
        self.r_l, self.r_u = r_l, r_u
        self.nr = nr
        self.rvals = linspace(r_l, r_u, nr)

        # Pass info to the base class initializer.
        super(PoissonRateInference, self).__init__(self.rvals, prior, self.lfunc)

    def lfunc(self, rvals):
        """
        Evaluate the Poisson likelihood function on a grid of rates.
        """
        r_intvl = self.intvl*rvals
        return (r_intvl)**self.n * exp(-r_intvl)

