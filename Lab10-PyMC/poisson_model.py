"""


Created Mar 27, 2015 by Tom Loredo
"""

import numpy as np
import scipy
from scipy import *
from scipy import stats
import pymc

from matplotlib.pyplot import *
try:
    import myplot
    from myplot import close_all, csavefig
    #myplot.tex_on()
    csavefig.save = False
except ImportError:
    pass


ion()

# A model factory function:

def make_poisson(n, intvl, mean0):
    """
    Make a PyMC model for inferring a Poisson distribution rate parameter,
    for a datum consisting of `n` counts observed in an interval of size
    `intvl`.  The inference will use an exponential prior for the rate,
    with prior mean `mean0`.
    """

    # PyMC's exponential dist'n uses beta = 1/scale = 1/mean.
    # Here we initialize rate to n/intvl.
    rate = pymc.Exponential('rate', beta=1./mean0, value=float(n)/intvl)

    # The expected number of counts, mu=rate*intvl, is a deterministic function
    # of the rate RV (and the constant intvl).
    @pymc.deterministic
    def mu(rate=rate):
        return rate*intvl

    # Poisson likelihood function:
    count = pymc.Poisson('count', mu=mu, value=n, observed=True)

    return locals()


if False:
    # Use MAP to find Max. A Posteriori estimate (i.e., mode):
    map = pymc.MAP(make_poisson(5, 1., 20.))
    map.fit()  # optimizes and saves results as attributes
    print 'MAP estimate of rate: {:.2f}'.format(float(map.rate.value))

    # Use NormApprox to fit a normal to the posterior, similar to the Laplace
    # approximation (but not exactly the same):
    normal = pymc.NormApprox(make_poisson(5, 1., 20.))
    normal.fit()  # optimizes and saves results as attributes

    # Use MCMC for posterior sampling via MCMC; store samples in a Python
    # pickle file:
    mcmc = pymc.MCMC(make_poisson(5, 1., 20.), db='pickle')
    # Run 3 chains:
    for i in range(3):
        mcmc.sample(iter=10000, burn=5000, thin=1)
    print  # to handle missing newline from progress bar

    # Generate a dict of Geweke test z scores for each RV, here using early
    # segments 10% of the chain length, a final segment 50% of the length,
    # and producing scores for 10 early intervals.
    scores = pymc.geweke(mcmc, first=0.1, last=0.5, intervals=10)

    # The Matplot functions automatically produce new figures for each plot.
    pymc.Matplot.geweke_plot(scores['rate'], 'rate')
    pymc.Matplot.geweke_plot(scores['mu'], 'mu')

    print 'Rhat values:', pymc.gelman_rubin(mcmc)

    # Plot credible regions and R values:
    pymc.Matplot.summary_plot(mcmc)


def make_on_off(n_off, expo_off, n_on, expo_on, mean0):
    """
    Make a PyMC model for inferring a Poisson signal rate parameter, `s`, for
    'on-off' observations with uncertain background rate, `b`.

    Parameters
    ----------

    n_off, n_on : int
        Event counts off-source and on-source

    expo_off, expo_on : float
        Exposures off-source and on-source

    mean0 : float
        Prior mean for both background and signal rates
    """

    # PyMC's exponential dist'n uses beta = 1/scale = 1/mean.
    # Here we initialize rates to good guesses.
    b_est = float(n_off)/expo_off
    s_est = max(float(n_on)/expo_on - b_est, .1*b_est)
    b = pymc.Exponential('b', beta=1./mean0, value=b_est)
    s = pymc.Exponential('s', beta=1./mean0, value=s_est)

    # The expected number of counts on and off source, as deterministic functions.
    @pymc.deterministic
    def mu_off(b=b):
        return b*expo_off

    @pymc.deterministic
    def mu_on(s=s, b=b):
        return (s+b)*expo_on

    # Poisson likelihood functions:
    off_count = pymc.Poisson('off_count', mu=mu_off, value=n_off, observed=True)
    on_count = pymc.Poisson('on_count', mu=mu_on, value=n_on, observed=True)

    return locals()


map = pymc.MAP(make_on_off(9, 1., 16, 1., 1000.))
map.fit()  # optimizes and saves results as attributes
print 'MAP estimate of signal rate: {:.2f}'.format(float(map.s.value))

mcmc = pymc.MCMC(make_on_off(9, 1., 16, 1., 1000.))
for i in range(3):
    mcmc.sample(iter=20000, burn=5000, thin=1)

pymc.Matplot.plot(mcmc)
pymc.Matplot.summary_plot(mcmc)

# mcmc.trace returns a database object; slice it to get the samples.
b_trace = mcmc.trace('b')[:]
s_trace = mcmc.trace('s')[:]

# Thin them so points for scatterplot are ~ independent.
b_trace = b_trace[0:-1:20]
s_trace = s_trace[0:-1:20]

# Use regular matplotlib for plotting this:
figure()
scatter(b_trace, s_trace, linewidths=0, alpha=.5)
xlabel('$b$ (counts/sec)')
xlim(xmin=0)
ylabel('$s$ (counts/sec)')
ylim(ymin=0)