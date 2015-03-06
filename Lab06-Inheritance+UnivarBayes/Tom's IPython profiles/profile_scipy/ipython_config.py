c = get_config()
app = c.InteractiveShellApp

# Merge default config into this one.
load_subconfig('ipython_config.py', profile='default')

lines = """
import numpy
import scipy
import matplotlib as mpl
from matplotlib.pyplot import *
from scipy import *
from scipy import stats, special, integrate, optimize, linalg, interpolate
from scipy import fftpack, signal, io, constants
import myplot
from myplot import close_all, csavefig
"""

# If there is a pre-existing exec_lines, append these; otherwise create it.
if hasattr(app, 'exec_lines'):
    app.exec_lines.append(lines)
else:
    app.exec_lines = [lines]

