import sys
import glob
import numpy
import pandas as pd
import matplotlib
#matplotlib.use('Agg') # -----(1)
import math
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

obsn = 'ONTA'

dirname='/home/vois/users/kanno/EVG/'

#call result of takagi2019
tfe = pd.read_csv(dirname+'takagi2019.csv')

for mtname in mtnames:
	mtn = mtname
	#parameters
	dstart = '2013-01-01 12:00'
	dend   = '2015-01-01 12:00'
	ax1ymax = []
	#gymax = 4 # for strain
	gymax = 30 # for raw
	rolnum = 3	
	shnum=[2,3,4]
	mdnum=[6,7,8,9,10,11]
	lgnum=[12,13,14,15,16,17,18,19,20]

	#shoff = 2 #for strain
	#lgoff = -2 # for strain
	
	shoff = 15
	lgoff = -15
	
	plotunit = 'raw' #strain or raw
	


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

	dfe  = dfe[dstart:dend]
	dfeB = dfeB[dstart:dend]
#figure plot

	fig = plt.figure(figsize=(6.8, 8.0),dpi=100)
	gs = fig.add_gridspec(4,1)
	ax1 = fig.add_subplot(gs[0,:])
	ax2 = fig.add_subplot(gs[1:4,:])
	
	ax12 = ax1.twinx()
	ax22 = ax2.twinx()

#data plot

	#ax1.plot(dfe,color="k",label="A+B")
	#ax1.plot(dfeB,color="b",label="B")

#if you want plot bar

	ax1.bar(dfe.index,dfe["Counts"],width=2,color='k',label="A+B type")
	ax1.bar(dfeB.index,dfeB["Counts"],width=2,color='g',label="B type")

	#calculate cumlative sum of number of EQ
	dfecum = numpy.cumsum(dfe)
	
	ax12.plot(dfecum,color="b",label='Cumulative')
	ax1.set_title(mtname+\
	' Earthquake')
	
	dxtime = pd.DataFrame(index=\
	[pd.to_datetime(dstart),pd.to_datetime(dend)])
	xmin = dxtime.index[0]
	xmax = dxtime.index[1]
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
			tfer.Erend[rnum], color="red",alpha=0.7,zorder=1)
			

	ax1.set_xlim([xmin,xmax])
	
	if ax1ymax:
		ax1.set_ylim([0,ax1ymax])
	
	ax12.set_ylim([0,ymax])
	ax1.set_ylabel('Number per day')
	ax12.set_ylabel('Cumulative number')
	
	ax1.grid()
	ax1.legend(bbox_to_anchor=(1.2,1),loc='upper left')
	ax12.legend(bbox_to_anchor=(1.2,0.5),loc='upper left')
	
#Load GNSS file names
	
	tmp = pd.read_csv(dirname+'stackname.csv')
	tmp = tmp.query('mtname == "'+mtname+'"')
	tmp = tmp.query('L > 1')
	#tmp = tmp.sort_values('L', ascending=False)
	tmp = tmp.reset_index()



	filenames = []
	dfgn = pd.DataFrame()
	dates_DF = pd.DataFrame\
	(index=pd.date_range(dstart,dend,freq='D'))
	dates_DF.index.name = 'Date'
	for inum in range(len(tmp)):
	
		tmpfilecode1 = tmp.loc[inum,['code1']].iloc[0]
		tmpfilecode2 = tmp.loc[inum,['code2']].iloc[0]
		tmpfilenames = glob.glob(dirname+\
		'GNSScsv02/*'+tmpfilecode1+'*'+tmpfilecode2+'*')
		dftmp = pd.read_csv(tmpfilenames[0],index_col=0,parse_dates=[0])
		dftmp.rename(columns={'dll(mm)': \
		tmpfilecode1+'-'+tmpfilecode2+' ('+\
		str(tmp.loc[inum,['L']].iloc[0])+'m)'},inplace=True)
		
		if plotunit == 'strain':
			dftmp = (dftmp*10**-3/tmp.loc[inum,['L']].iloc[0])*10**6
		else:
			dftmp = dftmp
		
		dftmp = dftmp - dftmp.mean()
		dfgn = pd.merge(dfgn,dftmp,how='outer',\
		left_index=True, right_index=True)	
		filenames.append(tmpfilenames[0])

	dfgn = dfgn[dstart:dend]
	dfgn = dfgn - dfgn.mean()
	dfgn = pd.merge(dfgn,dates_DF,how='outer',\
	left_index=True, right_index=True)
	dfgn = dfgn.rolling(rolnum,center=True).mean()


	#load tilt data
	print(mtn)
	print(obsn)
	dfti = pd.read_csv(dirname+'/'+mtn+'_'+obsn+'.csv',index_col=0,parse_dates=[0])
	
	print(dfti)

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

	dfti = dfti*mg
	dfti = dfti[dfti.index < ax1xmax]
	dfti = dfti-dfti.mean()

	dfC = dfti.copy()

	#noise reduction
	dfC = dfC.rolling(window=24).median() #24 hours median filter

	dfCD = dfC.copy()

	#gger step-like noise
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
 

#plot GNSS files
	
	#for fnum in range(len(filenames)):
		
	#	filename = filenames[fnum]
	#	df = pd.read_csv(filename,index_col=0,parse_dates=[0])
	#	
	#	labelname = tmp.loc[fnum,['codename1']].iloc[0]+\
	#	'-'+tmp.loc[fnum,['codename2']].iloc[0]+','+\
	#	str(tmp.loc[fnum,['L']].iloc[0])+'m'
	#
	#	ax2.plot(df-df['dll(mm)'].mean(),\
	#	ls='None',markersize=1,marker=".",\
	#	label=labelname,color=cm.jet(fnum/len(filenames)))
	#shnum=[3,4,5]
	#ax2.plot(dfgn,ls='None',markersize=1,marker=".")
	#ax2.plot(dfgn.mean(axis=1),'k',ls='None',markersize=3,marker=".")
	dfgnsh = dfgn.iloc[:,shnum] + shoff
	dfgnmd = dfgn.iloc[:,mdnum] 
	dfgnlg = dfgn.iloc[:,lgnum] + lgoff
	dfgnsh.plot(ax=ax2,ls='None',markersize=3,marker=".")
	dfgnsh.mean(axis=1).plot(ax=ax2,\
	ls='None',markersize=5,marker=".",color='k',label="SH stacked")
	
	dfgnmd.plot(ax=ax2,ls='None',markersize=3,marker=".")
	dfgnmd.mean(axis=1).plot(ax=ax2,\
	ls='None',markersize=5,marker=".",color='k',label="MD stacked")
	
	dfgnlg.plot(ax=ax2,ls='None',markersize=3,marker=".")
	dfgnlg.mean(axis=1).plot(ax=ax2,\
	ls='None',markersize=5,marker=".",color='k',label="LG stacked")
	
	
	
	ax2.set_title('GNSS data')
	ax2.set_xlim([xmin,xmax])
	ax2.set_xlabel('Time')
	if plotunit == 'strain':
		ax2.set_ylabel('Strain (10$^{-6}$)')
	else:
		ax2.set_ylabel('mm')
	

	ax2.set_ylim([-gymax,gymax])
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
	
	ax2.legend(bbox_to_anchor=(1.2,1),loc="upper left",fontsize=8)
	#ax2.legend()	
	
	ax22.plot(dfCD.index,dfCD['RD_C'],"k",linewidth=3,label="RADIDAL")	
	ax22.set_ylabel('$/mu$Rad')
	plt.tight_layout()
	ax22.legend(bbox_to_anchor=(1.2,0),loc="lower left",fontsize=8)

	#plt.savefig(dirname+'/Figs/stacked/figure'+mtname+'.png') # -----(2)
	#plt.close()
	#plt.clf() #------(3)
 
	plt.show()


