"""
Trivial example of a nose test case; run at command line with:

nosetests TrivialTest.py

or in an IPython notebook with:

!nosetests TrivialTest.py

Created Feb 20, 2015 by Tom Loredo
"""

import numpy as np
import scipy
import matplotlib as mpl
from matplotlib.pyplot import *
from scipy import *
from scipy import stats, special, integrate, optimize, linalg, interpolate
from scipy import fftpack, signal, io, constants

import myplot
from myplot import close_all, csavefig

ion()
#myplot.tex_on()
csavefig.save = False

a = 1.
b = 1.

plot([0,1,2],[0,1,4], 'b-', lw=2)

def test_ab():
    assert a == b
