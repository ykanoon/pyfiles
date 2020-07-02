#Modified by Yo Kanno, 2020/3/27
#if you cannot read Japanese in these files, enter as...
# :w
# then
# :edit ++encoding=utf-8



#plotgre.py
-plot GR-plot for 1 station, and many region.
-regions are defined by grepath.py
-save as GR_*stationname*.png

-for example
- > python plotgre.py grech.csv

	#grepath.py
	-parameter file for plotgre.py
	save csv files from "kensokuchi kensaku tool" in "path" directory

	#grech.csv
	-plot station ch name. you can add ch as many you want


#plotgrs.py
-plot GR-plot for many stations
-save as GRs_*.png

	#grspath.py
	-parameter file for plotgrs.py


#plotkensokuchi.py
-plot kensokuchi with x-y plot. youcan choose max amplitude ratio as 'ratio',
-'Max amp','S-P','P-P'




#myf.py
-functions 

