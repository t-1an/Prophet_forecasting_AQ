##########################################################################################
#    Script for plotting the change in HGV contributions to traffic volume in Manc       #
#                                                                                        #
#    This is free software: you can redistribute it and/or modify it under               #
#    the terms of the GNU General Public License as published by the Free Software       #
#    Foundation, either version 3 of the License, or (at your option) any later          #
#    version.                                                                            #
#                                                                                        #
#    This is distributed in the hope that it will be useful, but WITHOUT                 #
#    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS       #
#    FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more              #
#    details.                                                                            #
#                                                                                        #
#    You should have received a copy of the GNU General Public License along with        #
#    this repository.  If not, see <http://www.gnu.org/licenses/>.                       #
#                                                                                        #
##########################################################################################
# 2020, author David Topping: david.topping@manchester.ac.uk

import pyreadr
import os.path
import os
import requests
import pdb
import wget
import pandas as pd
import numpy as np
import datetime
import sys
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import datetime


# WHere is the traffic data stored?
frame_traff = pd.read_csv('pvr_2020-01-01_136d_city_type.csv')
metadata = pd.read_csv('GM_PERMANENT_ATC_sites.csv')
frame_traff['datetime'] = pd.to_datetime(frame_traff['Sdate'])
frame_traff = frame_traff.sort_values(by='datetime',ascending=True)
frame_traff=frame_traff.set_index('datetime')
#frame_traff=frame_traff.resample('60Min').sum()

# The selection of columns is based on the classes included on the Drakewell platform
frame_traff['type_sum']=(frame_traff['Class1Volume']+frame_traff['Class2Volume']+frame_traff['Class3Volume']+frame_traff['Class4Volume']+frame_traff['Class5Volume'])
frame_traff['non HGV']=(frame_traff['Class1Volume']+frame_traff['Class2Volume']+frame_traff['Class3Volume']+frame_traff['Class5Volume'])

# Define a mask for the COVID19 lockdown
mask_lockdown = (frame_traff.index > '2020-3-27')
mask_other = (frame_traff.index <= '2020-3-27')

frame_traff['lockdown'] = mask_lockdown

# This flag allows us to specify a list of sites to look at
manual_sites =  False

# Now we need to run through all of the sites and
# create different data frames according to site ID
Site_Names = frame_traff['Cosit'].unique().tolist()

data_dict = {name: frame_traff.loc[frame_traff['Cosit'] == name] for name in Site_Names}
# Set the index in each dict to datetime
for site in data_dict.keys():
    try:
        data_dict[site]['datetime'] = pd.to_datetime(data_dict[site]['Sdate'])
        data_dict[site]=data_dict[site].set_index('datetime')
        data_dict[site]=data_dict[site].resample('60Min').sum()
        data_dict[site]['HGV_ratio']=data_dict[site]['Class4Volume']/data_dict[site]['type_sum']
    except:
        print("Could not convert ", site)


################################################################################
# --------------------- Mean Daily volume of HGV -----------------------
fig, ax = plt.subplots(1, 2, figsize=(15, 10))
colormap = plt.cm.nipy_spectral
number_of_plots = len(data_dict.keys())
#colors = [colormap(i) for i in np.linspace(0, 1,number_of_plots)]
colors = sns.color_palette("hls", number_of_plots)
linewidth = np.linspace(1, 3,number_of_plots)
ax[0].set_prop_cycle(color=colors,lw=linewidth)
for site in data_dict.keys():

    mask_lockdown = (data_dict[site].index > pd.to_datetime('2020-1-20'))

    if manual_sites is True:
        if site in selected_sites:
        #pdb.set_trace()
            #mask_lockdown = (data_dict[site].index > pd.to_datetime('2020-1-20')) and (data_dict[site].index < pd.to_datetime('2020-5-15'))
            location_name = metadata[metadata['Site ID']==site]['Description'].values[0]
            ax[0].plot(data_dict[site].loc[mask_lockdown]['HGV_ratio'].resample('D').mean(),label=location_name)
    else:
        #pdb.set_trace()
        #mask_lockdown = (data_dict[site].index > pd.to_datetime('2020-2-28'))
        location_name = metadata[metadata['Site ID']==site]['Description'].values[0]
        ax[0].plot(data_dict[site].loc[mask_lockdown]['HGV_ratio'].resample('D').mean(),label=location_name)

fig.delaxes(ax[1])
ax[0].grid()
ax[1].grid(False)
plt.legend(loc="center right", borderaxespad=0.1,bbox_to_anchor=(2.2, 0.5))
plt.setp(ax[0].get_xticklabels(), rotation=90)
ax[0].set(xlabel='Date', ylabel='Mean daily HGV ratio')
ax[0].set_xlim([datetime.date(2020, 1, 20), datetime.date(2020, 5, 14)])
plt.subplots_adjust(bottom=0.25)
plt.show()
pdb.set_trace()
# --------------------- Mean Daily volume of HGV -----------------------
################################################################################

################################################################################
# --------------------  Individual site analysis -----------------------
# Now lets show the change in numbers of cars and HGVs as a diurnal profile
# at each site
for site in data_dict.keys():

    f, ax = plt.subplots(3,1,figsize=(15, 10))
    mask_lockdown = (data_dict[site].index > '2020-3-27')
    location_name = metadata[metadata['Site ID']==site]['Description'].values[0]
    plt.text(0.02, 0.9, location_name, fontsize=10, transform=plt.gcf().transFigure)
    data_dict[site]['lockdown'] = mask_lockdown

    if manual_sites is True:
        if site in selected_sites:
            sns.boxplot(data=data_dict[site],x=data_dict[site].index.hour, y=data_dict[site]['Class4Volume'],hue='lockdown', ax=ax[0]).set(xlabel='Hour of day',ylabel='HGV numbers')
            sns.boxplot(data=data_dict[site],x=data_dict[site].index.hour, y=data_dict[site]['non HGV'],hue='lockdown', ax=ax[1]).set(xlabel='Hour of day',ylabel='non HGV numbers')
            sns.boxplot(data=data_dict[site],x=data_dict[site].index.hour, y=data_dict[site]['HGV_ratio'],hue='lockdown', ax=ax[2]).set(xlabel='Hour of day',ylabel='HGV ratio')
    else:
        sns.boxplot(data=data_dict[site],x=data_dict[site].index.hour, y=data_dict[site]['Class4Volume'],hue='lockdown', ax=ax[0]).set(xlabel='Hour of day',ylabel='HGV numbers')
        sns.boxplot(data=data_dict[site],x=data_dict[site].index.hour, y=data_dict[site]['non HGV'],hue='lockdown', ax=ax[1]).set(xlabel='Hour of day',ylabel='non HGV numbers')
        sns.boxplot(data=data_dict[site],x=data_dict[site].index.hour, y=data_dict[site]['HGV_ratio'],hue='lockdown', ax=ax[2]).set(xlabel='Hour of day',ylabel='HGV ratio')

    plt.show()
    pdb.set_trace()
    plt.close('all')
