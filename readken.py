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
filename = 'futagoyamahokusei.csv'
mt = 'OSM'

regions = ['山頂付近','西方沖','北部','東部']
regionsE = ['Summit','West','North','East'] 

vshypfile = mt + '_hypo_list.csv' 

hypdf = myf.readhypvs(path + vshypfile)

marks = 'o+xvd^'

ch6 = 'miharayamahokusei'
ch7 = 'kitanoyama'
chx1='senduitonashi'
#chx1 = 'miharayamahokusei'
chx2='motomachi'
xptype = 'ratio' #ratio, S-P, Max amp
xscale = 'log' #log or linear
xlim = [0.01,100]

chy1='kitanoyama'
#chy1 = 'futagoyamahokusei'
yptype = 'ratio'
chy2='futagoyamahokusei'
yscale = 'log'
ylim = [0.01,100]

depthc = 'on' 
vmin = 0
vmax = 5

fig1 = plt.figure(figsize=(5,4))
ax1 = fig1.add_subplot(1,1,1)

num=0
for region in regions:
	
	if xptype == 'ratio':
		tmpX = myf.ratiodf(path,chx1,chx2,region)
		ax1.set_xlabel(chx1+' / '+chx2+' Max amp ratio')
	elif xptype == 'P-P':
		tmpX = myf.diffppdf(path,chx1,chx2,region)
		ax1.set_xlabel(chx1+' - '+chx2+' P-P')
	else:
		tmpX = myf.read_kensokuchicsv(path,chx1 + '.csv')
		tmpX = tmpX[tmpX['Remarks'].str.contains(region,na=False)]		
		ax1.set_xlabel(chx1+' '+xptype)


	if yptype == 'ratio':
		tmpY = myf.ratiodf(path,chy1,chy2,region)
		ax1.set_ylabel(chy1+' / '+chy2+' Max amp ratio')
	
	elif yptype == 'P-P':
		tmpY = myf.diffppdf(path,chy1,chy2,region)
		ax1.set_ylabel(chy1+' - '+chy2+' P-P')	
	
	else:
		tmpY = myf.read_kensokuchicsv(path,chy1 + '.csv')
		tmpY = tmpY[tmpY['Remarks'].str.contains(region,na=False)]	
		ax1.set_ylabel(chy1+' '+yptype)


	tmpXY = pd.merge(tmpX,tmpY,\
	how='outer',right_index=True,left_index=True)
	
	if depthc == 'on':
		tmpXY = pd.merge(tmpXY,hypdf['Depth'],
		how='outer',right_index=True,left_index=True)
		tmpXY.dropna(inplace=True)	
	
	#print(tmpXY)
	#exit()

	


	if xptype == yptype:
		if depthc == 'on':	
			ax1.scatter(tmpXY[xptype + '_x'],tmpXY[yptype + '_y'],c=tmpXY['Depth'],\
			marker=marks[num],label=regionsE[num],vmin=vmin,vmax=vmax,cmap=cm.jet)
		else:
			ax1.scatter(tmpXY[xptype + '_x'],tmpXY[yptype + '_y'],\
			marker=marks[num],label=regionsE[num])
	
	else:
		#ax1.plot(tmpXY[xptype],tmpXY[yptype],\
		#'o',markerfacecolor='None',markersize=5,label=regionsE[num])
		ax1.scatter(tmpXY[xptype + '_x'],tmpXY[yptype + '_y'],\
		marker=marks[num],label=regionsE[num])	


	
	ax1.set_xlim(xlim)
	ax1.set_ylim(ylim)
	ax1.set_yscale(xscale)
	ax1.set_xscale(yscale)
	
	num=num+1

ax1.set_title(xptype +'(X) vs '+ yptype+'(Y)')


norm = Normalize(vmin = vmin, vmax = vmax)

mappable = ScalarMappable(cmap = 'jet',norm=norm)
mappable._A=[]
fig1.colorbar(mappable).set_label('Depth (km)')


plt.tight_layout()
plt.grid()
plt.legend()

#plot GR-plot by myf.plotgr
#figtmp, axtmp = myf.plotgr(path,chGR,regions,regionsE,1400,(0,700))
#figtmp.savefig(savepath+'GR_'+chGR+'.png')

plt.show()


Xdata = 'S-P'
Ydata = 'S-P'

fig = plt.figure(figsize=(5,4))
ax3 = fig.add_subplot(1,1,1)
num=0
for region in regions:
	dfSP1 = myf.read_kensokuchicsv(path,ch6+'.csv')
	dfSP1 = dfSP1[dfSP1['Remarks'].str.contains(region,na=False)]
	dfSP2 = myf.read_kensokuchicsv(path,ch7+'.csv')
	dfSP2 = dfSP2[dfSP2['Remarks'].str.contains(region,na=False)]

	tmpXY = pd.merge(dfSP1[Xdata],dfSP2[Ydata],\
	how='outer',left_index=True,right_index=True)
	if Xdata == Ydata:
		ax3.plot(tmpXY[Xdata+'_x'],tmpXY[Ydata+'_y'],\
		'o',markerfacecolor='None',label=regionsE[num])	
	else:
		ax3.plot(tmpXY[Xdata],tmpXY[Ydata],\
		'o',markerfacecolor='None',label=regionsE[num])

	ax3.set_title(Xdata+' vs '+Ydata)
	ax3.set_xlabel(ch6+' '+Xdata)
	ax3.set_ylabel(ch7+' '+Ydata)
	num=num+1

print('Now plotting')

plt.grid()
plt.legend()
plt.show()
