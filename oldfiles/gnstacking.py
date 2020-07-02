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
#%matplotlib inline
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#this version try to plot stacking method
#mtnames=['ATS','MEA','TOK','TAR','KUT','USU','HKM','ESN','HKD','TWD','AKY','ZAO','AZM','ADT','BND','NAS','NSH','KSH','ASM','NYK','MDH','YAK','NOR','ONT','FUJ','HKN','ITV','OSM','MKJ','TRG','ASO','UNZ','KIR','SKR','KER']

mtnames=['ONT']

dirname='/home/vois/users/kanno/EVG/'

#call result of takagi2019
tfe = pd.read_csv(dirname+'takagi2019.csv')

for mtname in mtnames:
	
	#extruct 
	tfe2 = tfe.query('Mt == "'+mtname+'"')
	tfer = tfe.query('Mt == "'+mtname+'"')
	#skip NaN
	tfe2 = tfe2.dropna(subset=['GStart'])
	tfe2 = tfe2.reset_index()
	
	tfer = tfer.dropna(subset=['ErStart'])
	tfer = tfer.reset_index() 

	print('Now plotting '+mtname)

	#load Earthquake data
	#numbers for A+B earth quake
	dfe = pd.read_csv(dirname+'EQcsv/Vwx_graph_dsp_'+mtname+\
	'_AB.csv', skiprows=11,index_col=0,parse_dates=[0])

	dfe = dfe.drop(dfe.columns[[0,2,3]],axis=1)
	dfe.index.name = 'Date'	
	dfe.columns = ['Counts']
	
	#numbers for B type earthquake
	dfeB = pd.read_csv(dirname+'EQcsv/Vwx_graph_dsp_'+mtname+\
	'_B.csv', skiprows=11,index_col=0,parse_dates=[0])
	dfeB = dfeB.drop(dfeB.columns[[0,2,3]],axis=1)
	dfeB.index.name = 'Date'  
	dfeB.columns = ['Counts']

#figure plot

	fig = plt.figure(figsize=(12.0, 6.0))
	ax1 = fig.add_subplot(211)#for earthquake
	ax2 = fig.add_subplot(212)#for gnss data
	ax12 = ax1.twinx()


#data plot

	ax1.plot(dfe,color="k",label="A+B")
	ax1.plot(dfeB,color="b",label="B")

#if you want plot bar

	#ax1.bar(dfe.index,dfe["Counts"],width=2.0,color='k',label="A+B")
	#ax1.bar(dfeB.index,dfeB["Counts"],width=2.0,color='r',label="B")

	#calculate cumlative sum of number of EQ
	dfecum = numpy.cumsum(dfe)
	
	ax12.plot(dfecum,color="b")
	ax1.set_title(mtname+' EQ number and Total EQ number')

	xmin = dfe.index[0]
	xmax = dfe.index[-1]
	ymax = dfecum.max().iloc[0]
	ynmax = dfe.max().iloc[0]
	if tfe2.empty:
		print("tfe2 empty, no geodedic derformation detected")
	else:
		for gnum in range(len(tfe2)):
			ax1.axvspan(tfe2.GStart[gnum],\
			tfe2.GEnd[gnum], color="gray",alpha=0.3)
	
	if tfer.empty:
		print("No eruption")
	else:
		for rnum in range(len(tfer)):
			ax1.axvspan(tfer.ErStart[rnum],\
			tfer.Erend[rnum], color="red",alpha=0.7)
			

	ax1.set_xlim([xmin,xmax])
	ax1.set_ylim([0,100])
	ax12.set_ylim([0,ymax])
	ax1.set_ylabel('Number')
	ax12.set_ylabel('Total EQ number')
	
	ax1.grid(which="both")
	
	ax1.legend(loc='upper left')

	
#Load GNSS file names
	
	tmp = pd.read_csv(dirname+'stackname.csv')
	tmp = tmp.query('mtname == "'+mtname+'"')
	tmp = tmp.query('L > 1')
	tmp = tmp.sort_values('L', ascending=False)
	tmp = tmp.reset_index()
	
	print(tmp)

	tmp1 = pd.DataFrame({ "numbers": [2,3,1] }, index = [ pd.to_datetime("2018-2-14"), pd.to_datetime("2018-2-16"), pd.to_datetime("2018-2-17")])
	tmp1.index.name = "Date"
	print(tmp1)

	tmp2 = pd.DataFrame({ "numbers": [4,5,6] }, index = [ pd.to_datetime("2018-2-15"), pd.to_datetime("2018-2-18"), pd.to_datetime("2018-2-19")])
	tmp1.index.name = "Date"
		
	dates_DF1 = pd.DataFrame(index=pd.date_range('2018-2-14',periods=10,freq='D'))
	print(dates_DF1)
	
	tmp1.merge(tmp2)
	tmp0 = pd.DataFrame()
	tmp0 = pd.merge(tmp0,tmp1,how='outer',left_index=True, right_index=True)
	tmp0 = pd.merge(tmp0,tmp2,how='outer',left_index=True, right_index=True)	
	tmp0 = pd.merge(tmp0,dates_DF1,how='outer',left_index=True, right_index=True)
	print(tmp0)


	filenames = []
	dfgn = pd.DataFrame()
	dates_DF = pd.DataFrame(index=pd.date_range('2001-01-01 12:00','2019-11-01 12:00',freq='D'))
	dates_DF.index.name = 'Date'
	for inum in range(len(tmp)):
	
		tmpfilecode1 = tmp.loc[inum,['code1']].iloc[0]
		tmpfilecode2 = tmp.loc[inum,['code2']].iloc[0]
		tmpfilenames = glob.glob(dirname+\
		'GNSScsv02/*'+tmpfilecode1+'*'+tmpfilecode2+'*')
		#print(tmpfilenames)
		dftmp = pd.read_csv(tmpfilenames[0],index_col=0,parse_dates=[0])
		dftmp.rename(columns={'dll(mm)': tmpfilecode1+'-'+tmpfilecode2},inplace=True)
		print(dftmp)
		#print(dftmp.info())
		#print(dates_DF)
		#print(dates_DF.info())
		#print(len(dftmp))
		#print(len(dates_DF))
		dates_DF.merge( dftmp , how="outer", left_index=True, right_index=True)
		#print(len(dftmp))
		dfgn = pd.merge(dfgn,dftmp,how='outer',left_index=True, right_index=True)	
		filenames.append(tmpfilenames[0])

	dfgn = pd.merge(dfgn,dates_DF,how='outer',left_index=True, right_index=True)
	print(dfgn)

	exit()

#plot GNSS files
	for fnum in range(len(filenames)):
		
		filename = filenames[fnum]
		df = pd.read_csv(filename,index_col=0,parse_dates=[0])
		
		labelname = tmp.loc[fnum,['codename1']].iloc[0]+\
		'-'+tmp.loc[fnum,['codename2']].iloc[0]+','+\
		str(tmp.loc[fnum,['L']].iloc[0])+'m'

		ax2.plot(df-df['dll(mm)'].mean(),\
		ls='None',markersize=1,marker=".",\
		label=labelname,color=cm.jet(fnum/len(filenames)))

	ax2.set_title('GNSS data')
	ax2.set_xlim([xmin,xmax])
	ax2.set_xlabel('Time')
	ax2.set_ylabel('DL (mm)')
	ax2.legend(bbox_to_anchor=(1.01,1),\
	loc='upper left',borderaxespad=1,markerscale=12)
	ax2.set_ylim([-25,25])
	ax2.grid(which="both")

	if tfe2.empty:
                print("tfe2 empty!")
	else:
		for gnum in range(len(tfe2)):
                	ax2.axvspan(tfe2.GStart[gnum],\
			tfe2.GEnd[gnum], color="gray",alpha=0.3)

	if tfer.empty:
		print("No eruption!")
	else:
		for rnum in range(len(tfer)):
			ax2.axvspan(tfer.ErStart[rnum],\
			tfer.Erend[rnum], color="red",alpha=0.7)
		

	plt.tight_layout()

	#plt.savefig(dirname+'/Figs/stacked/figure'+mtname+'.png') # -----(2)
	plt.close()
	#plt.clf() #------(3)
 
	plt.show()

