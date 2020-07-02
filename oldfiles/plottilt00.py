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

dirname='/home/vois/users/kanno/EVG/'

df = pd.read_csv(dirname+'/ONT_ONTA.csv',index_col=0,parse_dates=[0])
#df = pd.read_csv(dirname+'/ONT_ONTA.csv',index_col=0)
#df.index = pd.to_datetime(df.index, format='%Y-%m-%d')

#print(df)


#figure prm
mg=10**6

ax1ymin=-0.00001*mg
ax1ymax=0.00001*mg

ax1xmin=datetime.datetime.strptime('2014-1-1','%Y-%m-%d')
ax1xmax=datetime.datetime.strptime('2017-1-1','%Y-%m-%d')

#print(ax1xmin)

df=df*mg

#plot figure
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(111,ylim=(ax1ymin,ax1ymax),xlim=(ax1xmin,ax1xmax))
#ax1 = fig.add_subplot(111)
#ax1.set_ylim([ax1ymin,ax1ymax])

#ax2 = fig.add_subplot(2,1,2)

ax1.plot(df.index,df['ONTA_EW_C'])

#df[['ONTA_EW_C','ONTA_NS_C']].plot(ax=ax1)
#df.plot(y='ONTA_NS_C',ax=ax1,xlim=[ax1xmin,ax1xmax],ylim=[ax1ymin,ax1ymax])
#ax1.set_ylim([-1,1])
#ax1.set_xlim([ax1xmin,ax1xmax])

plt.show()


