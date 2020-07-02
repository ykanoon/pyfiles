import sys
import numpy
import pandas as pd
import matplotlib
#matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt
from scipy import signal
import datetime
import matplotlib.dates as mdates
#%matplotlib inline
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#mtname='AZM'

mtnames=['AZM']

dirname='/home/vois/users/kanno/EVG/'

for mtname in mtnames:
	print(mtname)

	dfe = pd.read_csv(dirname+'EQcsv/Vwx_graph_dsp_'+mtname+'_AB.csv', skiprows=11,index_col=0,parse_dates=[0])

	dfe = dfe.drop(dfe.columns[[0,2,3]],axis=1)
	
	
	print(dfe.index[-1])

#	exit()

	dfeB = pd.read_csv(dirname+'EQcsv/Vwx_graph_dsp_'+mtname+'_B.csv', skiprows=11,index_col=0,parse_dates=[0])
	dfeB = dfeB.drop(dfeB.columns[[0,2,3]],axis=1)


	df = pd.read_csv(dirname+'GNSScsv/Gnh_15729399198298_J322-J327_DLL.csv',index_col=0,parse_dates=[0])


#figure draw

	fig = plt.figure(figsize=(12.0, 6.0))

	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	ax12 = ax1.twinx()
#data plot

	#sxmin='2013-01-01'
	#sxmin=dfe.index[0]
	sxmax='2019-06-01'



	ax1.plot(dfe,color="k")
	ax1.plot(dfeB,color="r")
	ax12.plot(numpy.cumsum(dfeB),color="b")
	ax1.set_title(mtname+' EQ number / Total EQ number')

	#xmin = datetime.datetime.strptime(sxmin, '%Y-%m-%d')
	xmin = dfe.index[0]
	xmax = dfe.index[-1]

	#ax1.set_yscale('log')

	ax1.set_xlim([xmin,xmax])
	ax1.set_ylim([0,100])

	ax1.set_xlabel('Time')
	ax1.set_ylabel('Number')
	ax1.grid(which="both")

	ax2.plot(df,ls='None',marker=".")
	ax2.set_title('GNSS data')
	ax2.set_xlim([xmin,xmax])
	ax2.set_xlabel('Time')
	ax2.set_ylabel('DL (mm)')

	ax2.grid(which="both")


	plt.tight_layout()

	#plt.savefig('figure'+mtname+'.png') # -----(2)
	#plt.clf() 
	plt.show()

