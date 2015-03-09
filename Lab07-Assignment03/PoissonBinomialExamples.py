"""
Plot Poisson rate posterior PDFs, and binomial alpha posterior PDFs, as
a demo of the UnivariateBayesianInference class.

Created 2015-03-06 by Tom Loredo
"""

import numpy as np
from numpy.testing import assert_approx_equal
import scipy
import matplotlib as mpl
from matplotlib.pyplot import *
from scipy import *
from scipy import stats, special, integrate

from poisson_binomial import PoissonRateInference, BinomialInference

# Optional plot customization module:
# import myplot
# from myplot import close_all
# 
# ion()
# #myplot.tex_on()



#-------------------------------------------------------------------------------
# 1st 2 curves:  Poisson, const & exp'l priors, (n,T) = (16, 2)

r_u = 20.  # upper limit for PDF calculation and plotting

# Flat prior case:
prior_l, prior_u = 0., 1e5
flat_pdf = 1./(prior_u - prior_l)
n, T = 16, 2
pri1 = PoissonRateInference(T, n, flat_pdf, r_u)
pri1.plot(alpha=.5)

# Exp'l prior:
scale = 10.
gamma1 = stats.gamma(1, scale=scale)  # a=1 is exp'l dist'n

pri2 = PoissonRateInference(T, n, gamma1.pdf, r_u)
pri2.plot(ls='g--')


xlabel(r'Rate (s$^{-1}$)')
ylabel('PDF (s)')


#-------------------------------------------------------------------------------
# 2nd 2 curves:  Binomial, const & beta(.5,.5) priors, (n, n_trials) = (8, 12)

# Define the data.
n, n_trials = 8, 12

bi1 = BinomialInference(n, n_trials)
bfig = figure()  # separate figure for binomial cases
bi1.plot(alpha=.5)

beta_half = stats.beta(a=.5, b=.5)
bi2 = BinomialInference(n, n_trials, beta_half.pdf, arange=(1.e-4, 1-1.e-4))
bi2.plot(ls='g--')

xlabel(r'$\alpha$')
ylabel('Posterior PDF')

