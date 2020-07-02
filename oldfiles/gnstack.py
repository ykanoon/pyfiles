import sys
import os
import glob
import numpy
import pandas as pd
import matplotlib

matplotlib.use('Agg') # -----(1)

import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import signal
from scipy import stats
import datetime
import matplotlib.dates as mdates
#%matplotlib inline
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import myf


#this version try to plot stacking method
#mtnames=['ATS','MEA','TOK','TAR','KUT','USU','HKM','ESN','HKD','TWD','AKY','ZAO','AZM','ADT','BND','NAS','NSH','KSH','ASM','NYK','MDH','YAK','NOR','ONT','FUJ','HKN','ITV','OSM','MKJ','TRG','ASO','UNZ','KIR','SKR','KER']

paranames=['MEA1','MEA2','MEA3','AZM1','AZM2','ONT1','ONT2','HKN1','HKN2','HKN3','HKN4']

frag=0 #if frag 1, skip depth plot

dirname='/Users/yokanno/Dropbox/JMA開発案件/kanno/EVG/'

#call result of takagi2019
tfe = pd.read_csv(dirname+'takagi2019.csv')

prmdf = pd.DataFrame\
({\
'ONT1':['ONT','ONTA','2013-07-01 12:00','2015-01-01 12:00',[],30,5,\
[2,3,4],[6,7,8,9,10,11],[12,13,14,15,16,17,18,19,20],15,-15],\
'ONT2':['ONT',[],'2006-01-01 12:00','2007-07-01 12:00',[],30,5,\
[2,3],[6,7,8,9,10,11],[12,13,14,15,16,17,18,19,20],15,-15],\
'HKN1':['HKN',[],'2012-01-01 12:00','2014-01-01 12:00',[],30,5,\
[0,1,2],[3,4,5,6,7,8],[9,10,11,12,13,14,15],15,-15],\
'HKN2':['HKN',[],'2014-01-01 12:00','2016-01-01 12:00',[],30,5,\
[0,1,2],[3,4,5,6,7,8],[9,10,11,12,13,14,15],15,-15],\
'HKN3':['HKN',[],'2016-01-01 12:00','2018-10-01 12:00',[],30,5,\
[0,1,2],[3,4,5,6,7,8],[9,10,11,12,13,14,15],15,-15],\
'HKN4':['HKN',[],'2018-01-01 12:00','2019-10-01 12:00',[],30,5,\
[0,1,2],[3,4,5,6,7,8],[9,10,11,12,13,14,15],15,-15],\
'MEA1':['MEA',[],'2014-01-01 12:00','2017-01-01 12:00',[],40,5,\
[0,1,2,3],[4,5,6],[7,8,9,10,11,12,13],20,-25],\
'MEA2':['MEA',[],'2017-01-01 12:00','2019-10-01 12:00',[],40,5,\
[0,1,2,3],[4,5,6],[7,8,9,10,11,12,13],20,-25],\
'MEA3':['MEA',[],'2014-01-01 12:00','2019-10-01 12:00',[],40,5,\
[0,1,2,3],[4,5,6],[7,8,9,10,11,12,13],20,-25],\
'AZM1':['AZM',[],'2013-09-01 12:00','2016-01-01 12:00',[],40,5,\
[0,1,2,4],[5,6,7],[8,9,10,11,12,13],20,-25],\
'AZM2':['AZM',[],'2017-11-01 12:00','2019-11-01 12:00',[],45,5,\
[0,1,2,4],[5,6,7],[8,9,10,11,12,13],20,-25]},\
index = ['mtn','obsn','dstart','dend','ax1ymax','gymax','rolnum','shnum',\
'mdnum','lgnum', 'shoff','lgoff'])



for paraname in paranames:
	
	mtname = prmdf.loc['mtn',[paraname]].iloc[0]
	print(mtname)
	mtn = prmdf.loc['mtn',[paraname]].iloc[0]
	
#parameters
	dstart = prmdf.loc['dstart',[paraname]].iloc[0]
	dend = prmdf.loc['dend',[paraname]].iloc[0]
	ax1ymax = []
	gymax = prmdf.loc['gymax',[paraname]].iloc[0] # for raw
	rolnum = 3	
	shnum = prmdf.loc['shnum',[paraname]].iloc[0]
	mdnum = prmdf.loc['mdnum',[paraname]].iloc[0]
	lgnum = prmdf.loc['lgnum',[paraname]].iloc[0]
	
	obsn = prmdf.loc['obsn',[paraname]].iloc[0]
	
	shoff = prmdf.loc['shoff',[paraname]].iloc[0]
	lgoff = prmdf.loc['lgoff',[paraname]].iloc[0]
	
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
	#dfe = myf.readvoisEQ(dirname+'EQcsv/Vwx_graph_dsp_'+mtname+'_AB.csv')	

	dfe = myf.readhypcEQ(dirname+'hypdspfiles/kanno_tmp/'+mtname+'de.csv')
		

	if(os.path.exists( dirname+'hypdspfiles/kanno_tmp/'+mtname+'dv.csv') ):
		dfetmp = myf.readhypcEQ( dirname+'hypdspfiles/kanno_tmp/'+mtname+'dv.csv'  )
	
		print(dfe)
		print(dfetmp)	

		dfe = pd.merge(dfe,dfetmp,how='outer',left_index=True, right_index=True)
		dfe.fillna(0,inplace=True)
		dfe['Counts'] = dfe['Counts_x']+dfe['Counts_y']
		dfe.drop(['Counts_x','Counts_y'], axis=1, inplace=True)
		print('merged')
		print(dfe)

	#exit()

#numbers for B type earthquake
	dfeB = myf.readvoisEQ(dirname+'EQcsv/Vwx_graph_dsp_'+mtname+'_B.csv')

	dfe  = dfe[dstart:dend]
	dfeB = dfeB[dstart:dend]

#load hypdsp data
	hyppath = dirname+'hypdspfiles/kanno_tmp/'+mtname+'e.csv'
	hypdf = myf.readhypEQ(hyppath)

	hyppathv = dirname+'hypdspfiles/kanno_tmp/'+mtname+'v.csv'

	if(os.path.exists(hyppathv)):
		#hypdfv = myf.readhypEQ(hyppathv)
		hypdf = hypdf.append(myf.readhypEQ(hyppathv)).sort_index()
	




#figure plot

	fig = plt.figure(figsize=(6.8, 8.0),dpi=100)
	gs = fig.add_gridspec(4,1)
	ax1 = fig.add_subplot(gs[0,:])
	axD = fig.add_subplot(gs[1,:])
	ax2 = fig.add_subplot(gs[2:4,:])
	
	ax12 = ax1.twinx()
	ax22 = ax2.twinx()

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

#plot hypdsp	

	if(os.path.exists(hyppath)):
		axD.plot(hypdf.index,hypdf['Depth(km)'],'o',Markersize=1)
		axD.set_xlim([xmin,xmax])
		axD.set_ylim([35,-3])
		axD.grid()
		axD.set_ylabel('Depth')
		axD.set_title('EQ Depth')
		axD.set_xticks([])

#Load GNSS file names
	
	tmp = pd.read_csv(dirname+'stackname.csv')
	tmp = tmp.query('mtname == "'+mtname+'"')
	tmp = tmp.query('L > 1')
	#tmp = tmp.sort_values('L', ascending=False)
	tmp = tmp.reset_index()

	print(tmp)

	filenames = []
	dfgn = pd.DataFrame()
	dates_DF = pd.DataFrame\
	(index=pd.date_range(dstart,dend,freq='D'))
	dates_DF.index.name = 'Date'
	for inum in range(len(tmp)):
	
		tmpfilecode1 = tmp.loc[inum,['code1']].iloc[0]
		tmpfilecode2 = tmp.loc[inum,['code2']].iloc[0]
		tmpfilenames = glob.glob(dirname+\
		'GNSScsv03/*'+tmpfilecode1+'*'+tmpfilecode2+'*')
		
	
		print(inum)
		print(tmpfilenames[0])		
		dftmp = pd.read_csv(tmpfilenames[0],index_col=0,parse_dates=[0])
		#dftmp.rename(columns={'dll(mm)': \
		#tmpfilecode1+'-'+tmpfilecode2+' ('+\
		#str(tmp.loc[inum,['L']].iloc[0])+'m)'},inplace=True)

		dftmp.rename(columns={'dll(mm)': \
		tmpfilecode1+'-'+tmpfilecode2},inplace=True)		


		if plotunit == 'strain':
			dftmp = (dftmp*10**-3/tmp.loc[inum,['L']].iloc[0])*10**6
		else:
			dftmp = dftmp
		
		dftmp = dftmp - dftmp.mean()
		
		dfgn = pd.merge(dfgn,dftmp,how='outer',\
		left_index=True, right_index=True)	
		filenames.append(tmpfilenames[0])

	
	dfgn = dfgn[dstart:dend]
	
	
	dfgn = pd.merge(dfgn,dates_DF,how='outer',\
	left_index=True, right_index=True)
	
	
	dfgn = dfgn - dfgn.mean()
	dfgn = dfgn.rolling(rolnum,center=True).mean()


#load tilt data
	print(mtn)
	print(obsn)
	
	if obsn:
	
		dfti = pd.read_csv(dirname+'/'+mtn+'_'+obsn+'.csv',index_col=0,parse_dates=[0])
		
		#print(dfti)

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

		print(dfCD)


#plot GNSS files
	
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
	

	if obsn:
		ax22.plot(dfCD.index,dfCD['RD_C'],"k",linewidth=3,label="RADIDAL")	
		ax22.set_ylabel(r'$\mu$rad')
		ax22.legend(bbox_to_anchor=(1.2,0),loc="lower left",fontsize=8)

	plt.tight_layout()


	plt.savefig(dirname+'/Figs/stacked/figure'+paraname+'.png') # -----(2)
	plt.close()
	plt.clf() #------(3)
 
	#plt.show()

