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

print(df)


#figure prm
mg=10**6

ax1ymin=-5
ax1ymax=5

ax1xmin=datetime.datetime.strptime('2011-1-1','%Y-%m-%d')
ax1xmax=datetime.datetime.strptime('2018-1-1','%Y-%m-%d')

df = df*mg
df = df[df.index < datetime.datetime(2018,1,1)]
df = df-df.mean()


dfC = df

#noise reduction
dfC = dfC.rolling(window=24).median()

#direction
dfCD = dfC
th = 2*pi - pi/4

NS=dfCD['ONTA_NS_C']*math.cos(th)
EW=dfCD['ONTA_EW_C']*math.sin(th)

#dfCD['ONTA_NS_C'] = dfCD['ONTA_NS_C']*math.cos(th)
#dfCD['ONTA_EW_C'] = dfCD['ONTA_EW_C']*math.sin(th)

#plot figure
fig = plt.figure(figsize=(8,6))

ax1 = fig.add_subplot(211,ylim=(ax1ymin,ax1ymax),xlim=(ax1xmin,ax1xmax))
ax2 = fig.add_subplot(212,ylim=(-0.5,0.5),xlim=(ax1xmin,ax1xmax))
#ax1.plot(df.index,df['ONTA_NS_C'],"k",label="ONTA_NS_C_raw")
#ax1.plot(df.index,df['ONTA_EW_C'],"k",label="ONTA_EW_C_raw")

#ax1.plot(dfC.index,dfC['ONTA_NS_C'],"r",label="ONTA_NS_C")
#ax1.plot(dfC.index,dfC['ONTA_EW_C'],"b",label="ONTA_EW_C")

ax1.plot(NS,"m",marker='.',label="ONTA_NS_CD")
ax1.plot(EW,"c",marker='.',label="ONTA_EW_CD")

ax2.plot(NS.diff())
ax2.plot(EW.diff())

plt.legend()
plt.show()


