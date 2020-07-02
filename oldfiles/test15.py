import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt
from scipy import signal
import datetime
import matplotlib.dates as mdates
#%matplotlib inline
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#mtname='AZM'

mtnames=['AZM','YKD']

for mtname in mtnames:
	print(mtname)

	dfe = pd.read_csv('/Users/yokanno/Dropbox/eqvsgd/EVG/EQcsv/Vwx_graph_dsp_'+mtname+'_AB.csv', skiprows=60,index_col=0,parse_dates=[0])

	dfe = dfe.drop(dfe.columns[[0,2,3]],axis=1)


	dfeB = pd.read_csv('/Users/yokanno/Dropbox/eqvsgd/EVG/EQcsv/Vwx_graph_dsp_'+mtname+'_B.csv', skiprows=60,index_col=0,parse_dates=[0])
	dfeB = dfeB.drop(dfeB.columns[[0,2,3]],axis=1)


	df = pd.read_csv('/Users/yokanno/Dropbox/eqvsgd/gnss/gnssdata/Gnh_15705461502588_J322-J327_DLL.csv',index_col=0,parse_dates=[0])


#figure draw

	fig = plt.figure(figsize=(12.0, 6.0))

	ax1= fig.add_subplot(211)
	ax2= fig.add_subplot(212)

#data plot

	sxmin='2013-01-01'
	sxmax='2019-06-01'



	ax1.plot(dfe,color="k")
	ax1.plot(dfeB,color="r")
	ax1.set_title(mtname+' EQ number / Total EQ number')

	xmin = datetime.datetime.strptime(sxmin, '%Y-%m-%d')
	xmax = datetime.datetime.strptime(sxmax, '%Y-%m-%d')

	ax1.set_yscale('log')

	ax1.set_xlim([xmin,xmax])
	ax1.set_ylim([0,200])

	ax1.set_xlabel('Time')
	ax1.set_ylabel('Number')

#ax1.set_yscale('log')

	ax1.grid(which="both")

	ax2.plot(df,ls='None',marker=".")
	ax2.set_title('GNSS data')
	ax2.set_xlim([xmin,xmax])
	ax2.set_xlabel('Time')
	ax2.set_ylabel('DL (mm)')

	ax2.grid(which="both")


	plt.tight_layout()

	plt.savefig('figure'+mtname+'.png') # -----(2)
	plt.clf() 
#plt.show()

