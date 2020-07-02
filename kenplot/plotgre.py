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
import grepath

args = sys.argv

print(args[1])

path = grepath.path
savepath = grepath.savepath
mt = grepath.mt

regions = grepath.regions
regionsE = grepath.regionsE

vshypfile = mt + '_hypo_list.csv' 
hypdf = myf.readhypvs(path + vshypfile)

marks = 'o+xvd^'
chs = pd.read_csv(args[1])

for numi in range(len(chs)):

	chGR = chs.loc[numi,['chGR']].iloc[0]
	rgmin = chs.loc[numi,['rgmin']].iloc[0]
	rgmax = chs.loc[numi,['rgmax']].iloc[0]	
	dltbin = chs.loc[numi,['dltbin']].iloc[0]
		
	#plot GR-plot by myf.plotgr
	figtmp, axtmp = myf.plotgre(path,chGR,regions,regionsE,dltbin,(rgmin,rgmax))
	figtmp.savefig(savepath+'GR_'+chGR+'.png')

	plt.tight_layout()


plt.show()

