import sys
import glob
import numpy
import pandas as pd
import matplotlib
#matplotlib.use('Agg') # -----(1)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import signal
import datetime
import matplotlib.dates as mdates
from matplotlib.dates import date2num
#%matplotlib inline
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import math
pi = math.pi

mtn = 'ONT'
obsn= 'ONTA'

dirname='/home/vois/users/kanno/EVG/'

df = pd.read_csv(dirname+'/'+mtn+'_'+obsn+'.csv',index_col=0,parse_dates=[0])

print(df)

#data prm
#params
NS = obsn+'_NS_C' #channel name of NS component
EW = obsn+'_EW_C' #channel name of EW component
td = 0.2 #threthould of diff
th = math.radians(310) #azimuth to vent

#figure prm
mg=10**6

ax1ymin=-5
ax1ymax=5

st='2011-1-1'
ed='2018-1-1'

ax1xmin=datetime.datetime.strptime(st,'%Y-%m-%d')
ax1xmax=datetime.datetime.strptime(ed,'%Y-%m-%d')

df = df*mg
df = df[df.index < ax1xmax]
df = df-df.mean()

dfC = df.copy()

#noise reduction
dfC = dfC.rolling(window=24).median() #24 hours median filter

dfCD = dfC.copy()

#trigger step-like noise
trgi = dfCD[ dfCD[EW].diff().abs()>td  ].index #diff over td
dfNS = dfCD[NS].diff()
dfEW = dfCD[EW].diff()

#remove step-like noise
for tnum in range(len(trgi)):
	#pntrint(trgi[tnum])
	#priantnt(dfNS.loc[ dfNS.index == trgi[tnum] ].iloc[0])
	#print( dfCD[ dfCD.index == trgi[tnum]  ].ONTA_NS_C )
	tmpNSdif = dfNS.loc[ dfNS.index == trgi[tnum] ].iloc[0] 
	tmpEWdif = dfEW.loc[ dfEW.index == trgi[tnum] ].iloc[0]
	dfCD.loc[ dfCD.index < trgi[tnum] , NS] = \
	dfCD.loc[ dfCD.index < trgi[tnum] , NS] + tmpNSdif
	dfCD.loc[ dfCD.index < trgi[tnum] , EW] = \
	dfCD.loc[ dfCD.index < trgi[tnum] , EW] + tmpEWdif

#direction correction
dfCD[NS] = dfCD[NS]*math.cos(th)
dfCD[EW] = dfCD[EW]*math.sin(th)
dfCD = dfCD - dfCD.mean()
dfCD['RD_C'] = (dfCD[NS]+dfCD[EW])/2



#plot figure
fig = plt.figure(figsize=(8,6))

ax1 = fig.add_subplot(111,ylim=(ax1ymin,ax1ymax),xlim=(ax1xmin,ax1xmax))
#ax2 = fig.add_subplot(212,ylim=(0,0.5),xlim=(ax1xmin,ax1xmax))

ax1.plot(df.index,df[NS],"k",label=NS+"_raw")
ax1.plot(df.index,df[EW],"k",label=EW+"_raw")

ax1.plot(dfC.index,dfC[NS],"r",label=NS+"_median_filterd")
ax1.plot(dfC.index,dfC[EW],"b",label=EW+"_median_filterd")

ax1.plot(dfCD.index,dfCD[NS],"m",label=NS+"_rm_step_dir_cor")
ax1.plot(dfCD.index,dfCD[EW],"c",label=EW+"_rm_step_dir_cor")
ax1.plot(dfCD.index,dfCD['RD_C'],"k",linewidth=3,label="RADIDAL")

ax1.set_title(mtn+' tilt data')
ax1.set_xlabel('Time')
ax1.set_ylabel("$\mu$rad")

#ax2.plot(dfCD.index,dfCD[EW].diff().abs(),"c",label="ONTA_EW_CD")
#ax2.plot(dfCD.index,dfCD[NS].diff().abs(),"m",label="ONTA_NS_CD")
#ax2.plot(dfCD.index,dfCD['ONTA_EW_C'].diff().abs(),"c",label="ONTA_EW_CD")


ax1.legend(bbox_to_anchor=(0,-0.1),loc='upper left')
#ax2.legend()

plt.tight_layout()
plt.show()


