import sys
import glob
import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import signal
import datetime
import matplotlib.dates as mdates
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import codecs
import myf
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import grspath

args = sys.argv


path = grspath.path
savepath = grspath.savepath
mt = grspath.mt

regions = grspath.regions
regionsE = grspath.regionsE

vshypfile = mt + '_hypo_list.csv' 
hypdf = myf.readhypvs(path + vshypfile)

marks = 'o+xvd^'

#chnames = ['miharayamahokusei','kitanoyama']
chnames = grspath.chnames

rgmin = grspath.rgmin
rgmax = grspath.rgmax
dltbin = grspath.dltbin
		
#plot GR-plot by myf.plotgr
figtmp, axtmp = myf.plotgrs(path,mt,chnames,dltbin,(rgmin,rgmax))
figtmp.savefig(savepath+'GRs_'+mt+'.png')

plt.tight_layout()

plt.show()

