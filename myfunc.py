#modifile 2020/6/16 by Yo Kanno
import pandas as pd
import os
import codecs
import matplotlib.pyplot as plt
from pyfiles import myfunc
import numpy
import glob
import math
import datetime
import calendar
import statsmodels.api as sm


def prmdf2prm(prmdf,paraname):
	

	mtname = prmdf.loc['mtn',[paraname]].iloc[0]
	#print(mtname)
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


	return(mtname, mtn, dstart, dend, ax1ymax, gymax, rolnum, shnum, mdnum, lgnum, obsn, shoff, lgoff, plotunit)


def extfe(tfe,mtname):
	"""
	extruct data from tfe (eruption and geodedic data from table in Takagi 2019)
	
	Return
	---------------
	tfer: dataframe
		Eruption date data
	tfe2: dataframe
		Geodetic deformation date data
	
	"""
	print('Start importing from tfe, by extfe')
	#extruct 
	tfe2 = tfe.query('Mt == "'+mtname+'"')
	tfer = tfe.query('Mt == "'+mtname+'"')

	#skip NaN
	tfe2 = tfe2.dropna(subset=['GStart'])
	tfe2 = tfe2.reset_index()

	tfer = tfer.dropna(subset=['ErStart'])
	tfer = tfer.reset_index()

	return(tfe2, tfer)




def readvoisEQ(voisEQpath):
	voisEQdf = pd.read_csv(voisEQpath, skiprows=11,index_col=0,parse_dates=[0])
	voisEQdf = voisEQdf.drop(voisEQdf.columns[[0,2,3]],axis=1)
	voisEQdf.index.name = 'Date'
	voisEQdf.columns = ['Counts']
	return(voisEQdf)

#read count/day data
def readhypcEQ(hcpath):
	"""
	Name:readhypcEQ
	
	Function
	import earthquake number per day data from hypdsp data
	read hypdisp "c"ount EQ	

	Authour: Yo Kanno
	Date:2020.6.16

	Parameters
	----------------
	hcpath : str
		path for EQnumber data,
		mtname+'de.csv' for data from epos
		mtname+'dv.csv' for data from hypcenter determined by vois  

	Returns
	----------------
	hypEDdf : dataframe
		index:Date
		key:
	"""
	hypEQdf = pd.read_csv(hcpath)
	hypEQdf.columns = ['y','m','d','Counts']
	hypEQdf['y'] = hypEQdf['y'].astype(str)
	hypEQdf['m'] = hypEQdf['m'].astype(str)
	hypEQdf['d'] = hypEQdf['d'].astype(str)
	hypEQdf['Date'] = hypEQdf['y'].str.cat(hypEQdf['m'],sep='-').str.cat(hypEQdf['d'],sep='-')
	
	hypEQdf.set_index('Date',inplace=True)
	hypEQdf = hypEQdf.drop(['y','m','d'], axis=1)
	hypEQdf.index = pd.to_datetime(hypEQdf.index)


	#print(hypEQdf.index.dtype)
	return(hypEQdf)

def EQnumberperdayHyp(dirhyp,mtname,dstart,dend):
	"""
	Name:EQnumberperdayHyp

	Function
	see also readhypcEQ 

	Authour: Yo Kanno
	Date:2020.6.16

	Parameters
	----------------
	dirhyp : str
		path for EQnumber data
	mtname : str
		mount name 3 char, AZM, ONT...

		mtname+'de.csv' for data from epos
		mtname+'dv.csv' for data from hypcenter determined by vois  
	

	Returns
	----------------
	dfe : dataframe
		A+B type EQ number per day
	dfecum : dataframe
		cumlative sum of number of EQ
	
	"""
	dfe = myfunc.readhypcEQ(dirhyp+mtname+'de.csv')

	if(os.path.exists( dirhyp+mtname+'dv.csv') ):
		dfetmp = myfunc.readhypcEQ( dirhyp+mtname+'dv.csv'  )

		dfe = pd.merge(dfe,dfetmp,how='outer',left_index=True, right_index=True)
		dfe.fillna(0,inplace=True)
		dfe['Counts'] = dfe['Counts_x']+dfe['Counts_y']
		dfe.drop(['Counts_x','Counts_y'], axis=1, inplace=True)	
	
	dfe  = dfe[dstart:dend]
	
	dfecum = numpy.cumsum(dfe)
	
	return(dfe, dfecum)


def EQnumberperdayVOIS(dirvois,mtname,dstart,dend):
	"""
	see also readvoisEQ
	this function enhance input parameters only
	"""
	dfeB = myfunc.readvoisEQ(dirvois+mtname+'_B.csv')

	dfeB = dfeB[dstart:dend]

	return(dfeB)




#read hypdsp data convert to data frame
def readhypEQ(hyppath):
	if(os.path.exists(hyppath)):
		print('Start loding hypdsp data')
		with codecs.open(hyppath, 'r', 'utf-8', 'ignore') as f:
			tmphyp = pd.read_csv(f)
		tmphyp.columns = ['rawdata']
		hypdf = pd.DataFrame()
		hypdf['Date'] = tmphyp['rawdata'].str[1:15]
		hypdf['Date'] = pd.to_datetime(hypdf['Date'])
		hypdf['Lat(m)'] = tmphyp['rawdata'].str[21:24].astype(int)
		hypdf['Lat(s)'] = tmphyp['rawdata'].str[24:28].astype(float)*10**-2
		hypdf['Lon(m)'] = tmphyp['rawdata'].str[32:36].astype(int)
		hypdf['Lon(s)'] = tmphyp['rawdata'].str[36:40].astype(float)*10**-2
		hypdf['Depth(km)'] = tmphyp['rawdata'].str[44:49].astype(float)*10**-2
		hypdf = hypdf.set_index('Date')
		return hypdf

def makehypdf(dirhyp,mtname):
	"""
	see also readhypEQ
	this function merge data from epos and vois
	"""
	hyppath = dirhyp+mtname+'e.csv'
	hypdf = myfunc.readhypEQ(hyppath)

	hyppathv = dirhyp+mtname+'v.csv'

	if(os.path.exists(hyppathv)):
		#hypdfv = myfunc.readhypEQ(hyppathv)
		hypdf = hypdf.append(myfunc.readhypEQ(hyppathv)).sort_index()


	return(hypdf)
	




#read vois hyp data from shingen list 
def readhypvs(vshyppath):
	#print('Start loding vois hyp data')
	with codecs.open(vshyppath, 'r', 'euc_jp', 'ignore') as f:
		tmphyp = pd.read_csv(f)
	tmphyp.columns = ['event ID','ID','flag','data type','event','region',\
	'OT','lat(h)','lat(s)','lat-o','lon(h)','lon(m)','lon-o','Depth','Depth-o',\
	'M','obs num','remarks']
	tmphyp['Start time'] = tmphyp['event ID' ].astype(str).str[3:-2]
	tmphyp['Start time'] = pd.to_datetime(tmphyp['Start time'])
	tmphyp.set_index('Start time', inplace = True)
	hypdf = tmphyp
	return hypdf


def makeGNSSdata(dirname,mtname,dstart,dend,rolnum,shnum,mdnum,lgnum,shoff,lgoff,plotunit):
	"""
	this function prepare GNSS data 	

	"""

	tmp = pd.read_csv(dirname+'stackname.csv')
	tmp = tmp.query('mtname == "'+mtname+'"')
	tmp = tmp.query('L > 1')
	#tmp = tmp.sort_values('L', ascending=False)
	tmp = tmp.reset_index()

	#print(tmp)

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


		#       print(inum)
		#       print(tmpfilenames[0])          
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


	dfgnsh = dfgn.iloc[:,shnum] + shoff
	dfgnmd = dfgn.iloc[:,mdnum]
	dfgnlg = dfgn.iloc[:,lgnum] + lgoff


	return(dfgn, dfgnsh, dfgnmd, dfgnlg)



def getGNSSpos(dirname,obs):

	df = pd.read_excel(dirname+'【最新版】観測点情報(GPS).xls', sheet_name='地理院マップシート用観測点',header=1)
	
	
	pos = df[ df['RINEX'] == obs ]
	
	
	pos = pos[ ['緯度（deg）', '経度(deg)'] ].values 
	
	return(pos)



#https://qiita.com/damyarou/items/9cb633e844c78307134a
def cal_rho(lon_a,lat_a,lon_b,lat_b):
	ra=6378.140  # equatorial radius (km)
	rb=6356.755  # polar radius (km)
	F=(ra-rb)/ra # flattening of the earth
	rad_lat_a=numpy.radians(lat_a)
	rad_lon_a=numpy.radians(lon_a)
	rad_lat_b=numpy.radians(lat_b)
	rad_lon_b=numpy.radians(lon_b)
	pa=numpy.arctan(rb/ra*numpy.tan(rad_lat_a))
	pb=numpy.arctan(rb/ra*numpy.tan(rad_lat_b))
	xx=numpy.arccos(numpy.sin(pa)*numpy.sin(pb)+numpy.cos(pa)*numpy.cos(pb)*numpy.cos(rad_lon_a-rad_lon_b))
	c1=(numpy.sin(xx)-xx)*(numpy.sin(pa)+numpy.sin(pb))**2/numpy.cos(xx/2)**2
	c2=(numpy.sin(xx)+xx)*(numpy.sin(pa)-numpy.sin(pb))**2/numpy.sin(xx/2)**2
	dr=F/8*(c1-c2)
	rho=ra*(xx+dr)
	return rho



def makestkGNSS(dirname,dstart,dend,dets,dete,rolnum,BLcodes,plotunit):

	dfgn = pd.DataFrame()
	dates_DF = pd.DataFrame(index=pd.date_range(dstart,dend,freq='D'))
	dates_DF.index.name = 'Date'

	for codes in BLcodes:
		tmpfilenames = glob.glob(dirname+'GNSScsv03/*'+codes[0]+'*'+codes[1]+'*')
		if not tmpfilenames:
			tmpfilenames = glob.glob(dirname+'GNSScsv03/*'+codes[1]+'*'+codes[0]+'*')

		dftmp = pd.read_csv(tmpfilenames[0],index_col=0,parse_dates=[0])

		cname = codes[0]+'-'+codes[1]

		dftmp.rename(columns={'dll(mm)': \
		cname},inplace=True)    

		dftmp = dftmp - dftmp.mean()
	
		if plotunit == 'strain':
			pos1 = myfunc.getGNSSpos(dirname,codes[0])
			pos2 = myfunc.getGNSSpos(dirname,codes[1])
			rho = myfunc.cal_rho(pos1[0,1],pos1[0,0],pos2[0,1],pos2[0,0])		
			print(rho)
	
			dftmp = (dftmp*10**-3/ ( rho*10**3 ) )*10**6



		#dftmp['trend'] = numpy.arange(0,len(dftmp))
		#dftmp['trend'] = dftmp['trend']*(-0.04/365)
		#dftmp[ cname ] = dftmp[ cname ] - dftmp['trend']			
		#dftmp.drop('trend',axis=1)
		print(type(dftmp))



		dfgn = pd.merge(dfgn,dftmp,how='outer',\
		left_index=True, right_index=True)

	dfgn = dfgn[dstart:dend]


	dfgn = pd.merge(dfgn,dates_DF,how='outer',\
	left_index=True, right_index=True)


	#dfgn = dfgn - dfgn.mean()
	dfgn = dfgn.rolling(rolnum,center=True).mean()
	dfgn = dfgn - dfgn.mean()
	
	
	

	#dfgn = dfgn.where( dfgn.diff().diff(-1).abs() < dfgn.std() )	



	return(dfgn)



def detrenddaydf(df,dets,dete):

	tm = df
	dtdf = df
	
	tmp = df[dets:dete].interpolate('linear').dropna()
	
	x = numpy.arange(0,len(tmp))
	y = tmp.values	
	
	ab = numpy.polyfit(x,y,1)
	

	dtdf['trend'] = ab[0]*numpy.arange(0,len(df))

	#dtdf['detrend'] =  

	return(dtdf)





def leapdaysnum(year):

	if calendar.isleap(year):
		ydays = 366 
	else:
		ydays = 365 

	return(ydays)




def yeardayday(yearday):

	deci = yearday - math.floor(yearday)
	
	dt0 = datetime.datetime( math.floor(yearday) ,1,1)	

	yd = deci*leapdaysnum(math.floor(yearday))
	
	ydd = dt0 + datetime.timedelta(days= round(yd) )
	
	return(ydd)






def makedgTgn(dirTKS,mtn,dstart,dend):
	""" 
	Parameters
	---------------
	dirTKS : str
	dir name for Takagi2019 stacked data
	mtn : str
	mt name 3 char, ONT, AZM etc.
	dstart : 

	"""         
	#dgTgn = pd.read_table(dirTKS+mtn+'_stackdtr.smo',header=None,delim_whitespace=True, index_col=0)
	
	if not os.path.isfile(dirTKS+mtn+'_stackdtr.smo'):
		print('No file for:'+dirTKS+mtn+'_stackdtr.smo')
		dgTgn = pd.DataFrame(index=['Date'], columns=['Strain(10^-6)'])
		return(dgTgn)
	
	dgTgn = pd.read_table(dirTKS+mtn+'_stackdtr.smo',header=None,delim_whitespace=True)
	
	
	#dgTgn.index.name = 'Date'
	#dgTgn.columns = ['Strain(10^-6)']
	dgTgn.columns = ['Date','Strain(10^-6)']
	
	for index, row in dgTgn.iterrows():
		dgTgn.at[index, 'Date'] =  \
		yeardayday( dgTgn.at[index, 'Date'] ) + \
		datetime.timedelta(hours=12)

	dgTgn.set_index('Date',inplace=True)

	dgTgn = dgTgn[dstart:dend]

	dates_DF = pd.DataFrame\
	(index=pd.date_range(dstart,dend,freq='D'))
	dates_DF.index.name = 'Date'
	dgTgn = pd.merge(dgTgn,dates_DF,how='outer',\
	left_index=True, right_index=True)

	return(dgTgn)





def makeTiltdata(dirname, mtn, obsn, td, thd, st, ed):
	"""
a	
	Parameters
	----------
	obsn : str
		Observation station name
	td : float
		threthould of diff
	thd : float
		azimuth to vent
	st : yyyy-MM-dd
		start date
	ed : yyyy-MM-dd
		end date

	"""
	if obsn:
		dfti = pd.read_csv(dirname+'/'+mtn+'_'+obsn+'.csv',index_col=0,parse_dates=[0])

		#print(dfti)

		#data prm
		#params
		NS = obsn+'_NS_C' #channel name of NS component
		EW = obsn+'_EW_C' #channel name of EW component
		td = 0.2 #threthould of diff
		th = math.radians(thd) #azimuth to vent

		#figure prm
		mg=10**6

		#st='2011-1-1'
		#ed='2018-1-1'

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

		#print(dfCD)
		return(dfCD)

	else:
		pass











#read kensokuchi kensaku tool data
def read_kensokuchicsv(path,filename):

	fin = codecs.open(path+filename,"r","euc_jp")
	fout_utf=codecs.open(path+'tmp.csv',"w","utf-8")
	for row in fin:
		fout_utf.write(row)
	fin.close()
	fout_utf.close()
	df = pd.read_csv(path+'tmp.csv')
	df.columns=['ID','Start time','End time','Flag','Type',\
	'Event','Region','Logical sensor','P time','S time','S-P',\
	'X','F','X-F','I','Direction','Max amp','Period','Unit','Remarks']
	df['Start time'] = pd.to_datetime(df['Start time'])
	df['End time'] = pd.to_datetime(df['End time'])
	df['P time'] = pd.to_datetime(df['P time'],format='%H:%M:%S.%f')
	df['S time'] = pd.to_datetime(df['S time'],format='%H:%M:%S.%f')
	df.set_index('Start time', inplace = True)
	return df


#calcurate max amp ratio from kensokuchi data
def ratiodf(path,ch1,ch2,region):
	df1 = read_kensokuchicsv(path,ch1+'.csv')
	df1 = df1[df1['Remarks'].str.contains(region,na=False)]

	df2 = read_kensokuchicsv(path,ch2+'.csv')
	df2 = df2[df2['Remarks'].str.contains(region,na=False)]

	tmp1 = df1['Max amp']
	tmp1 = tmp1[tmp1 != 'S.O.']
	tmp1 = tmp1.astype('f8')

	tmp2 = df2['Max amp']
	tmp2 = tmp2[tmp2 != 'S.O.']
	tmp2 = tmp2.astype('f8')

	tmp12 = pd.merge(tmp1,tmp2,how='outer',right_index=True,left_index=True)

	tmp12['ratio'] = tmp12['Max amp_x']/tmp12['Max amp_y']
	tmp12.drop('Max amp_x',axis=1,inplace=True)
	tmp12.drop('Max amp_y',axis=1,inplace=True)
	tmp12.dropna(inplace=True)
	return tmp12

def diffppdf(path,ch1,ch2,region):
	df1 = read_kensokuchicsv(path,ch1+'.csv')
	df1 = df1[df1['Remarks'].str.contains(region,na=False)]

	df2 = read_kensokuchicsv(path,ch2+'.csv')
	df2 = df2[df2['Remarks'].str.contains(region,na=False)]
	
	tmp1 = df1['P time']
	tmp2 = df2['P time']

	tmp12 = pd.merge(tmp1,tmp2,how='outer',right_index=True,left_index=True)
	tmp12['P-P'] = tmp12['P time_x']-tmp12['P time_y']
	tmp12['P-P'] = tmp12['P-P']/numpy.timedelta64(1, 's')
	tmp12.drop('P time_x',axis=1,inplace=True)
	tmp12.drop('P time_y',axis=1,inplace=True)
	tmp12.dropna(inplace=True)
	
	return tmp12


#plot cumulative earthquake number to show GR figure
def plotgre(path, ch, regions, regionsE, dltbin, rangegr):
	
	fig2 = plt.figure(figsize=(7,5))
	ax2 = fig2.add_subplot(1,1,1)
	num=0
	for region in regions:

		print('Now culc '+region)
		if region == 'No remarks':
			dfcum1 = myf.read_kensokuchicsv(path,ch+'.csv')
			dfcum1 = dfcum1[dfcum1['Remarks'].isnull()]
		elif region == 'All plot':
			dfcum1 = myf.read_kensokuchicsv(path,ch+'.csv')
		else:
			dfcum1 = myf.read_kensokuchicsv(path,ch+'.csv')
			dfcum1 = dfcum1[dfcum1['Remarks'].str.contains(region,na=False)]
		
		data = numpy.array(dfcum1['Max amp'].dropna())
		
		tmpbins = int ( round( (rangegr[1]-rangegr[0])/dltbin ) )


		hist, bin_edges = numpy.histogram(data, \
		bins = tmpbins ,range=rangegr)
		
		hist_df = pd.DataFrame(columns=['Start','End','Count'])
		for idx, val in enumerate(hist):
			start = round(bin_edges[idx],2)
			end = round(bin_edges[idx+1],2)
			hist_df.loc[idx] = [start,end,val]
		hist_df.sort_values('Start', ascending=False, inplace=True)
		hist_df['Cum count'] = numpy.cumsum(hist_df['Count'])
		hist_df.sort_values('Start',  inplace=True)

		ax2.plot(hist_df['Start'],hist_df['Cum count'],\
		'o',markerfacecolor='None',markersize=4,\
		label=regionsE[num]+' (n='+str(len(data))+')')
		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_title(ch + ' Cumulative Counts')
		ax2.set_xlabel(r'Max amplitude ($\mu$m/s) $\Delta$='+str(dltbin))
		ax2.set_ylabel('Cumulative counts')
		num=num+1

	plt.grid()
	plt.legend(bbox_to_anchor=(1.05,1),loc='upper left', borderaxespad=0)

	return fig2, ax2

def plotgrs(path, mt, chnames, dltbin, rangegr):
	
	
	fig2 = plt.figure(figsize=(7,5))
	ax2 = fig2.add_subplot(1,1,1)
	num=0

	
	for ch in chnames:	
		print(ch)

		dfcum1 = myf.read_kensokuchicsv(path,ch+'.csv')
		tmp = dfcum1['Max amp']
		tmp = tmp[tmp != 'S.O.'].astype('f8')

		data = numpy.array(tmp.dropna())

		tmpbins = int ( round( (rangegr[1]-rangegr[0])/dltbin ) )

		hist, bin_edges = numpy.histogram(data, \
		bins = tmpbins ,range=rangegr)

		hist_df = pd.DataFrame(columns=['Start','End','Count'])
		for idx, val in enumerate(hist):
			start = round(bin_edges[idx],2)
			end = round(bin_edges[idx+1],2)
			hist_df.loc[idx] = [start,end,val]
		hist_df.sort_values('Start', ascending=False, inplace=True)
		hist_df['Cum count'] = numpy.cumsum(hist_df['Count'])
		hist_df.sort_values('Start',  inplace=True)

		ax2.plot(hist_df['Start'],hist_df['Cum count'],\
		'o',markerfacecolor='None',markersize=4,\
		label=ch+' (n='+str(len(data))+')')
		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_title(mt + ' Cumulative Counts')
		ax2.set_xlabel(r'Max amplitude ($\mu$m/s) $\Delta$='+str(dltbin))
		ax2.set_ylabel('Cumulative counts')
		num=num+1

	plt.grid()
	plt.legend(bbox_to_anchor=(1.05,1),loc='upper left', borderaxespad=0)

	return fig2, ax2


#make tmpX(orY)
def maketmpXY(path,chx1,chx2,region,xptype):
	if xptype == 'ratio':
		tmpX = myf.ratiodf(path,chx1,chx2,region)
		tmplabel = chx1+' / '+chx2+' Max amp ratio'
	elif xptype == 'P-P':
		tmpX = myf.diffppdf(path,chx1,chx2,region)
		tmplabel = chx1+' - '+chx2+' P-P (s)'
	elif xptype == 'S-P':
		tmp = myf.read_kensokuchicsv(path,chx1 + '.csv')
		tmp = tmp[tmp['Remarks'].str.contains(region,na=False)]
		tmpX = tmp['S-P']
		tmplabel = chx1+' '+xptype+' (s)'
	elif xptype == 'Max amp' :
		tmp = myf.read_kensokuchicsv(path,chx1 + '.csv')
		tmp = tmp[tmp['Remarks'].str.contains(region,na=False)]
		tmpX = tmp['Max amp']
		tmpX = tmpX[tmpX != 'S.O.']
		tmpX = tmpX.astype('f8')
		tmplabel = chx1+' '+xptype+' (micro m/s)'
	
	return tmplabel, tmpX












