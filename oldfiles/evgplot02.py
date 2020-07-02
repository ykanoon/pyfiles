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


mtnames=['AZM']

dirname='/home/vois/users/kanno/EVG/'

for mtname in mtnames:

	print('Now plotting '+mtname)

	dfe = pd.read_csv(dirname+'EQcsv/Vwx_graph_dsp_'+mtname+\
	'_AB.csv', skiprows=11,index_col=0,parse_dates=[0])

	dfe = dfe.drop(dfe.columns[[0,2,3]],axis=1)

	dfeB = pd.read_csv(dirname+'EQcsv/Vwx_graph_dsp_'+mtname+\
	'_B.csv', skiprows=11,index_col=0,parse_dates=[0])
	
	dfeB = dfeB.drop(dfeB.columns[[0,2,3]],axis=1)

#figure draw

	fig = plt.figure(figsize=(12.0, 6.0))

	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	ax12 = ax1.twinx()

#data plot

	ax1.plot(dfe,color="k")
	ax1.plot(dfeB,color="r")
	ax12.plot(numpy.cumsum(dfeB),color="b")
	ax1.set_title(mtname+' EQ number / Total EQ number')

	xmin = dfe.index[0]
	xmax = dfe.index[-1]

	ax1.set_xlim([xmin,xmax])
	ax1.set_ylim([0,100])
	#ax12.set_ylim([0,inf])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('Number')
	ax1.grid(which="both")

	filenames=['Gnh_15729399198298_J322-J327_DLL.csv',\
			'Gnh_15729399198298_J322-J328_DLL.csv']
	
	for filename in filenames:

		print(filename)
		exit()
		df = pd.read_csv(dirname+\
        	'GNSScsv/'+filename,index_col=0,parse_dates=[0])

		ax2.plot(df-df['dll(mm)'].mean(),ls='None',marker=".")
		ax2.set_title('GNSS data')
		ax2.set_xlim([xmin,xmax])
		ax2.set_xlabel('Time')
		ax2.set_ylabel('DL (mm)')

	ax2.grid(which="both")


	plt.tight_layout()

	#plt.savefig('figure'+mtname+'.png') # -----(2)
	#plt.clf()
 
	plt.show()

