import sys
import glob
import numpy
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

#missing 'YAK','IWK','AKY','IWT','AKK'

mtnames=['ATS','MEA','TOK','TAR','KUT','USU','HKM','ESN','HKD','TWD','ZAO','AZM','ADT','BND','NAS','NSH','KSH','ASM','NYK','MDH','NOR','ONT','FUJ','HKN','ITV','OSM','MKJ','TRG','ASO','UNZ','KIR','SKR','KER']

#mtnames=['OSM']

dirname='/Users/yok/Dropbox/eqvsgd/EVG/'

for mtname in mtnames:

	print('Now plotting '+mtname)

	dfe = pd.read_csv(dirname+'EQcsv/Vwx_graph_dsp_'+mtname+\
	'_AB.csv', skiprows=11,index_col=0,parse_dates=[0])

	dfe = dfe.drop(dfe.columns[[0,2,3]],axis=1)
	dfe.index.name = 'Date'	
	dfe.columns = ['Counts']


	dfeB = pd.read_csv(dirname+'EQcsv/Vwx_graph_dsp_'+mtname+\
	'_B.csv', skiprows=11,index_col=0,parse_dates=[0])
	
	dfeB = dfeB.drop(dfeB.columns[[0,2,3]],axis=1)
	dfeB.index.name = 'Date'  
	dfeB.columns = ['Counts']

#figure draw

	fig = plt.figure(figsize=(12.0, 6.0))

	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	ax12 = ax1.twinx()

#data plot

	ax1.plot(dfe,color="k",label="A+B")
	ax1.plot(dfeB,color="r",label="A")
	#ax1.bar(dfe.index,dfe["Counts"],width=2.0,color='k',label="A+B")
	#ax1.bar(dfeB.index,dfeB["Counts"],width=2.0,color='r',label="B")	
	dfecum = numpy.cumsum(dfe)
	
	print(dfecum.max().iloc[0])
	
	ax12.plot(dfecum,color="b")
	ax1.set_title(mtname+' EQ number / Total EQ number')

	xmin = dfe.index[0]
	xmax = dfe.index[-1]
	ymax = dfecum.max().iloc[0]

	ax1.set_xlim([xmin,xmax])
	ax1.set_ylim([0,100])
	ax12.set_ylim([0,ymax])
	#ax1.set_xlabel('Time')
	ax1.set_ylabel('Number')
	ax12.set_ylabel('Total EQ number')
	ax1.grid(which="both")


	ax1.legend(loc='upper left')

	

#Load GNSS file names

	tmp = pd.read_csv(dirname+'DLname2.csv')
	tmp = tmp.query('mtname == "'+mtname+'"')
	tmp = tmp.query('L > 1')
	tmp = tmp.sort_values('L', ascending=False)
	tmp = tmp.reset_index()

	print(tmp)

	filenames = []

	for inum in range(len(tmp)):		
	
		tmpfilecode1 = tmp.loc[inum,['code1']].iloc[0]
		tmpfilecode2 = tmp.loc[inum,['code2']].iloc[0]
		tmpfilenames = glob.glob(dirname+'GNSScsv/*'+tmpfilecode1+'*'+tmpfilecode2+'*')	
		
		filenames.append(tmpfilenames[0])
	
	print(len(filenames))

	for fnum in range(len(filenames)):
		print(fnum)
		filename = filenames[fnum]

		print(filename[0:-4])
		
		df = pd.read_csv(filename,index_col=0,parse_dates=[0])

		#ax2.plot(df-df['dll(mm)'].mean(),ls='None',markersize=1,marker=".",label=filename[-17:-8])
		labelname = tmp.loc[fnum,['codename1']].iloc[0]+'-'+tmp.loc[fnum,['codename2']].iloc[0]+', '+str(tmp.loc[fnum,['L']].iloc[0])+'m'
		ax2.plot(df-df['dll(mm)'].mean(),ls='None',markersize=1,marker=".",label=labelname)
		ax2.set_title('GNSS data')
		ax2.set_xlim([xmin,xmax])
		ax2.set_xlabel('Time')
		ax2.set_ylabel('DL (mm)')

	ax2.legend(bbox_to_anchor=(1.01,1),loc='upper left',borderaxespad=1,markerscale=12)
	ax2.set_ylim([-100,100])
	ax2.grid(which="both")


	plt.tight_layout()

	plt.savefig('figure'+mtname+'.png') # -----(2)
	plt.clf()
 
	#plt.show()

