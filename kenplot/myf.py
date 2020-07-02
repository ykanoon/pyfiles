import pandas as pd
import os
import codecs
import matplotlib.pyplot as plt
import myf
import numpy

def readvoisEQ(voisEQpath):
	voisEQdf = pd.read_csv(voisEQpath, skiprows=11,index_col=0,parse_dates=[0])
	voisEQdf = voisEQdf.drop(voisEQdf.columns[[0,2,3]],axis=1)
	voisEQdf.index.name = 'Date'
	voisEQdf.columns = ['Counts']
	return(voisEQdf)

#read count/day data
def readhypcEQ(hcpath):
	hypEQdf = pd.read_csv(hcpath)
	hypEQdf.columns = ['y','m','d','Counts']
	hypEQdf['y'] = hypEQdf['y'].astype(str)
	hypEQdf['m'] = hypEQdf['m'].astype(str)
	hypEQdf['d'] = hypEQdf['d'].astype(str)
	hypEQdf['Date'] = hypEQdf['y'].str.cat(hypEQdf['m'],sep='-').str.cat(hypEQdf['d'],sep='-')
	
	hypEQdf.set_index('Date',inplace=True)
	hypEQdf = hypEQdf.drop(['y','m','d'], axis=1)
	hypEQdf.index = pd.to_datetime(hypEQdf.index)

	print(hypEQdf.index.dtype)
	return(hypEQdf)


#read hypdips data convert to data frame
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

#read vois hyp data from shingen list 
def readhypvs(vshyppath):
	print('Start loding vois hyp data')
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














