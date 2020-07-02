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

dirname='/home/vois/users/kanno/EVG/'

df = pd.read_csv(dirname+'/ONT_ONTA.csv',index_col=0,parse_dates=[0])
#df = pd.read_csv(dirname+'/ONT_ONTA.csv',index_col=0)
#df.index = pd.to_datetime(df.index, format='%Y-%m-%d')

#print(df)


#figure prm
mg=10**6

ax1ymin=-5
ax1ymax=5

ax1xmin=datetime.datetime.strptime('2011-1-1','%Y-%m-%d')
ax1xmax=datetime.datetime.strptime('2018-1-1','%Y-%m-%d')

df = df*mg
df = df[df.index < datetime.datetime(2018,1,1)]
df = df-df.mean()

NS = 'ONTA_NS_C'

dfC = df

#noise reduction
dfC = dfC.rolling(window=24).median()

#direction
dfCD = dfC.copy()
th = 2*pi - pi/4
td = 0.1

trgi = dfCD[ dfCD['ONTA_NS_C'].diff().abs()>td  ].index
dfab = dfCD[['ONTA_NS_C']].diff()

print( len(trgi) )

for tnum in range(len(trgi)):
	print(trgi[tnum])
	print(dfab.loc[ dfab.index == trgi[tnum] ].iloc[0])
	print( dfCD[ dfCD.index == trgi[tnum]  ].ONTA_NS_C )
	dfCD.loc[ dfCD.index < trgi[tnum] , 'ONTA_NS_C'] = dfCD.loc[ dfCD.index < trgi[tnum] , 'ONTA_NS_C'] + dfab.loc[ dfab.index == trgi[tnum] ].iloc[0] 



#dfCD['ONTA_NS_C'] = dfCD['ONTA_NS_C']*math.cos(th)
#dfCD['ONTA_EW_C'] = dfCD['ONTA_EW_C']*math.sin(th)
dfCD = dfCD - dfCD.mean()
dfCD['RD_C'] = (dfCD['ONTA_NS_C']+dfCD['ONTA_EW_C'])/2



#plot figure
fig = plt.figure(figsize=(8,6))

ax1 = fig.add_subplot(211,ylim=(ax1ymin,ax1ymax),xlim=(ax1xmin,ax1xmax))
ax2 = fig.add_subplot(212,ylim=(0,0.5),xlim=(ax1xmin,ax1xmax))

#ax1.plot(df.index,df['ONTA_NS_C'],"k",label="ONTA_NS_C_raw")
#ax1.plot(df.index,df['ONTA_EW_C'],"k",label="ONTA_EW_C_raw")

#ax1.plot(dfC.index,dfC['ONTA_NS_C'],"r",label="ONTA_NS_C")
#ax1.plot(dfC.index,dfC['ONTA_EW_C'],"b",label="ONTA_EW_C")

ax1.plot(dfCD.index,dfCD['ONTA_NS_C'],"m",label="ONTA_NS_CD")
ax1.plot(dfCD.index,dfCD['ONTA_EW_C'],"c",label="ONTA_EW_CD")
ax1.plot(dfCD.index,dfCD['RD_C'],"k",label="RADIDAL")

ax2.plot(dfCD.index,dfCD['ONTA_EW_C'].diff().abs(),"c",label="ONTA_EW_CD")
ax2.plot(dfCD.index,dfCD['ONTA_NS_C'].diff().abs(),"m",label="ONTA_NS_CD")
#ax2.plot(dfCD.index,dfCD['ONTA_EW_C'].diff().abs(),"c",label="ONTA_EW_CD")


plt.legend()
plt.show()


