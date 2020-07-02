import pandas as pd
import os
import codecs

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



