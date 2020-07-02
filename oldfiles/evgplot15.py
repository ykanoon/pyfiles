import sys
import glob
import numpy
import pandas as pd
import matplotlib
matplotlib.use('Agg') # -----(1)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import signal
import datetime
import matplotlib.dates as mdates
#%matplotlib inline
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#this version plot close up based on plotterm.csv

#missing 'YAK','IWK','AKY','IWT','AKK'
#ok 'YAK','AKY'
#csv error 'IWK': low activity
#csv erro  'IWT': if you choose ivent, NaN csv

#mtnames=['MEA','TAR','ZAO','AZM','KSH','ASM','NYK','ONT','HKN','MKJ','ASO','KIR','KER']

mtnames=['MEA']

dirname='/home/vois/users/kanno/EVG/'

#call result of takagi2019
tfe = pd.read_csv(dirname+'takagi2019.csv')
pltmts = pd.read_csv(dirname+'plotterm.csv')

print(pltmts)

print(tfe)

for mtname in mtnames:
	
	#extruct 
	tfe2 = tfe.query('Mt == "'+mtname+'"')
	tfer = tfe.query('Mt == "'+mtname+'"')
	#skip NaN
	tfe2 = tfe2.dropna(subset=['GStart'])
	tfe2 = tfe2.reset_index()
	""	
	if tfe2.empty:
		print("tfe2 empty!")
	else:
		print(tfe2)	
	""

	#tfer = tfe.query('Mt == "'+mtname+'"')
	tfer = tfer.dropna(subset=['ErStart'])
	tfer = tfer.reset_index() 

	if tfer.empty:
		print("No eruption!")
	else:
		print(tfer)

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


	terms = pltmts.query('Mt == "'+mtname+'"')
	terms = terms.reset_index()
	print(terms)
	
	for tenum in range(len(terms)):

	#figure draw
		fig =	 plt.figure(figsize=(12.0, 6.0))

		ax1 = fig.add_subplot(211)
		ax2 = fig.add_subplot(212)
		
	#data plot

		ax1.plot(dfe,color="k",label="A+B")
		ax1.plot(dfeB,color="b",label="B")

	#if you want plot bar

		#ax1.bar(dfe.index,dfe["Counts"],width=2.0,color='k',label="A+B")
		#ax1.bar(dfeB.index,dfeB["Counts"],width=2.0,color='r',label="B")	

		dfecum = numpy.cumsum(dfe)
		
		#print(dfecum.max().iloc[0])
		
		
		ax1.set_title(mtname+' EQ number / Total EQ number')

		
		xmin = terms.start[tenum]
		xmax = terms.end[tenum]

		ymax = dfecum.max().iloc[0]
		ynmax = dfe[xmin:xmax].max().iloc[0]
		if tfe2.empty:
			print("tfe2 empty!")
		else:
			for gnum in range(len(tfe2)):
				ax1.axvspan(tfe2.GStart[gnum],tfe2.GEnd[gnum], color="gray",alpha=0.3)
		
		if tfer.empty:
			print("No eruption!")
		else:
			for rnum in range(len(tfer)):
				ax1.axvspan(tfer.ErStart[rnum],tfer.Erend[rnum], color="red",alpha=0.3)			

		ax1.set_xlim([xmin,xmax])
		ax1.set_ylim([0,ynmax])
		#ax1.set_ylim(bottom=0)
		
		#ax1.set_xlabel('Time')
		ax1.set_ylabel('Number')
		
		
		ax1.grid(which="both")
		

		ax1.legend(loc='upper left')

		
	#Load GNSS file names
		
		tmp = pd.read_csv(dirname+'DLname2.csv')
		tmp = tmp.query('mtname == "'+mtname+'"')
		tmp = tmp.query('L > 1')
		tmp = tmp.sort_values('L', ascending=False)
		tmp = tmp.reset_index()

		#print(tmp)

		filenames = []

		for inum in range(len(tmp)):		
		
			tmpfilecode1 = tmp.loc[inum,['code1']].iloc[0]
			tmpfilecode2 = tmp.loc[inum,['code2']].iloc[0]
			tmpfilenames = glob.glob(dirname+'GNSScsv/*'+tmpfilecode1+'*'+tmpfilecode2+'*')	
		
			filenames.append(tmpfilenames[0])
		
		#print(len(filenames))

	#plot GNSS files
		#ax2.plot(tfer.ErStart[0],0,ls='None',marker=".",label="Eruption")
		for fnum in range(len(filenames)):
			#print(fnum)
			filename = filenames[fnum]

			#print(filename[0:-4])
			
			df = pd.read_csv(filename,index_col=0,parse_dates=[0])
			
			labelname = tmp.loc[fnum,['codename1']].iloc[0]+'-'+tmp.loc[fnum,['codename2']].iloc[0]+','+str(tmp.loc[fnum,['L']].iloc[0])+'m'

			ax2.plot(df-df['dll(mm)'].mean(),ls='None',markersize=3,marker=".",label=labelname,color=cm.jet(fnum/len(filenames)))
			ax2.plot(df-df['dll(mm)'].mean(),color=cm.jet(fnum/len(filenames)))
	 
		
		ax2.set_title('GNSS data')
		ax2.set_xlim([xmin,xmax])
		ax2.set_xlabel('Time')
		ax2.set_ylabel('DL (mm)')
		ax2.legend(bbox_to_anchor=(1.01,1),loc='upper left',borderaxespad=1,markerscale=12)
		ax2.set_ylim([-80,80])
		ax2.grid(which="both")

		if tfe2.empty:
			print("tfe2 empty!")
		else:
			for gnum in range(len(tfe2)):
				ax2.axvspan(tfe2.GStart[gnum],tfe2.GEnd[gnum], color="gray",alpha=0.3)

		if tfer.empty:
			print("No eruption!")
		else:
			for rnum in range(len(tfer)):
				ax2.axvspan(tfer.ErStart[rnum],tfer.Erend[rnum], color="red",alpha=0.3)
			

		plt.tight_layout()

		plt.savefig(dirname+'Figs/closeup/figure'+mtname+str(tenum)+'close.png') # -----(2)
		plt.clf() #------(3
	 
		#plt.show()

