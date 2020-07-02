import sys
import glob
import numpy
import pandas as pd
import matplotlib
#matplotlib.use('Agg') # -----(1)^M
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import signal
import datetime
import matplotlib.dates as mdates
#%matplotlib inline
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import codecs
import myf
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


path = '/home/vois/users/kanno/kensokuchi/'
savepath = path + 'Figs/'

mt = 'OSM'

# :w, then :edit ++encoding=utf-8
regions = ['山頂付近']
regionsE = ['Summit'] 

vshypfile = mt + '_hypo_list.csv' 

hypdf = myf.readhypvs(path + vshypfile)

marks = 'o+xvd^'

prms = pd.read_csv('figparams00.csv')


for numi in range(len(prms)):

	print('Now make figure... ',str(numi+1)+'/'+str(len(prms)))

	chx1=prms.loc[numi,['chx1']].iloc[0]
	chx2=prms.loc[numi,['chx2']].iloc[0]
	xptype = prms.loc[numi,['xptype']].iloc[0]  #ratio, S-P, Max amp
	xscale = prms.loc[numi,['xscale']].iloc[0]  #log or linear
	xlim = (prms.loc[numi,['xlim0']].iloc[0],prms.loc[numi,['xlim1']].iloc[0])

	print(xscale)
	print(xlim)	

	chy1=prms.loc[numi,['chy1']].iloc[0]
	chy2=prms.loc[numi,['chy2']].iloc[0]
	yptype = prms.loc[numi,['yptype']].iloc[0]  #ratio, S-P, Max amp
	yscale = prms.loc[numi,['yscale']].iloc[0]  #log or linear
	ylim = (prms.loc[numi,['ylim0']].iloc[0],prms.loc[numi,['ylim1']].iloc[0])

	print(yscale)
	print(ylim)

	depthc = prms.loc[numi,['depthc']].iloc[0]
	vmin = prms.loc[numi,['vmin']].iloc[0]
	vmax = prms.loc[numi,['vmax']].iloc[0]

	alptmp = 0

	fig1 = plt.figure(figsize=(5,4))
	ax1 = fig1.add_subplot(1,1,1)

	num=0
	#plot each region
	for region in regions:
		
		tmplabelX, tmpX = myf.maketmpXY(path,chx1,chx2,region,xptype)
		ax1.set_xlabel(tmplabelX)

		tmplabelY, tmpY = myf.maketmpXY(path,chy1,chy2,region,yptype)
		ax1.set_ylabel(tmplabelY)
		
		tmpXY = pd.merge(tmpX,tmpY,\
		how='outer',right_index=True,left_index=True)
		
		tmpXY.dropna(inplace = True)
		
		if depthc == 'on':
			tmpXY = pd.merge(tmpXY,hypdf['Depth'],
			how='outer',right_index=True,left_index=True)
			
		
		if xptype == yptype:
			if depthc == 'on':	
				ax1.scatter(tmpXY[xptype + '_x'],tmpXY[yptype + '_y'],c=tmpXY['Depth'],facecolors='none',\
				marker=marks[num],label=regionsE[num],vmin=vmin,vmax=vmax,cmap=cm.jet,alpha=alptmp,edgecolors="blue")
			else:
				ax1.scatter(tmpXY[xptype + '_x'].astype('f8'),tmpXY[yptype + '_y'],facecolors='none',\
				marker=marks[num],alpha = alptmp,label=regionsE[num],edgecolors="blue")
		else:
			if depthc == 'on':
				ax1.scatter(tmpXY[xptype],tmpXY[yptype],c=tmpXY['Depth'],facecolors='none',\
				marker=marks[num],label=regionsE[num],vmin=vmin,vmax=vmax,cmap=cm.jet,alpha = alptmp,edgecolors="blue")
			else:
				ax1.scatter(tmpXY[xptype],tmpXY[yptype],facecolors='none',\
				marker=marks[num],alpha = alptmp,label=regionsE[num],edgecolors="blue")

		ax1.set_xlim(xlim)
		ax1.set_ylim(ylim)
		ax1.set_xscale(xscale)
		ax1.set_yscale(yscale)
		
		num=num+1

	ax1.set_title(xptype +'(X) vs '+ yptype+'(Y)')

	mappable = ScalarMappable(cmap = 'jet',norm=Normalize(vmin = vmin, vmax = vmax))
	mappable._A=[]

	if depthc == 'on':
		fig1.colorbar(mappable).set_label('Depth (km)')

	plt.tight_layout()
	plt.grid()
	plt.legend()
	fig1.savefig(path+'Figs/ken'+str(numi)+'.png')	

plt.show()

