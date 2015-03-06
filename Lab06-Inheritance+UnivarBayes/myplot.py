from matplotlib.pyplot import *
# from matplotlib.figure import Figure
from matplotlib import rc

__all__ = ['tex_on', 'tex_off', 'close_all', 'csavefig']

rc('figure.subplot', bottom=.125, top=.95, right=.95)  # left=0.125
rc('font', size=14)  # default for labels (not axis labels)
rc('font', family='serif')  # default for labels (not axis labels)
rc('axes', labelsize=18)
rc('xtick.major', pad=8)
rc('xtick', labelsize=14)
rc('ytick.major', pad=8)
rc('ytick', labelsize=14)
rc('savefig', dpi=150)
rc('axes.formatter', limits=(-4,4))


# Turn TeX processing of labels on/off.

def tex_on():
    rc('text', usetex=True)
    rc('font',**{'family':'serif','serif':['Computer Modern Roman']})

def tex_off():
    rc('text', usetex=False)
    rc('font',**{'family':'serif','serif':['Times']})

def close_all():
    """
    Close all open figures.
    """
    close('all')
