##########################################################################################
#    Script for plotting the change in HGV contributions to traffic volume at Sharston   #
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

# Data from ATC near Sharston, on Style Road
frame_traff = pd.read_csv('pvr_2020-01-01_135d.csv')
metadata = pd.read_csv('GM_JOURNEY_TIME_sites.csv')

frame_traff['datetime'] = pd.to_datetime(frame_traff['Sdate'])
frame_traff = frame_traff.sort_values(by='datetime',ascending=True)
#pdb.set_trace()
frame_traff=frame_traff.set_index('datetime')
frame_traff=frame_traff.resample('60Min').sum()

frame_traff['type_sum']=(frame_traff['Class1Volume']+frame_traff['Class2Volume']+frame_traff['Class3Volume']+frame_traff['Class4Volume']+frame_traff['Class5Volume']+frame_traff['Class6Volume']+frame_traff['Class7Volume']+frame_traff['Class8Volume']+frame_traff['Class9Volume']+frame_traff['Class10Volume'])
frame_traff['non HGV']=(frame_traff['Class1Volume']+frame_traff['Class2Volume']+frame_traff['Class3Volume']+frame_traff['Class5Volume']+frame_traff['Class6Volume']+frame_traff['Class7Volume']+frame_traff['Class8Volume']+frame_traff['Class9Volume']+frame_traff['Class10Volume'])
frame_traff['HGV_ratio']=frame_traff['Class4Volume']/frame_traff['type_sum']

# Define a mask for the COVID19 lockdown
mask_lockdown = (frame_traff.index > '2020-3-27')
mask_other = (frame_traff.index <= '2020-3-27')

frame_traff['lockdown'] = mask_lockdown

################################################################################
# ---------------------Diurnal Profile Ratio of HGV ----------------------------
f, ax = plt.subplots(1,1,figsize=(12, 5))
sns.boxplot(data=frame_traff,x=frame_traff.index.hour, y=frame_traff['HGV_ratio'],hue='lockdown').set(xlabel='Hour of day',ylabel='HGV ratio')
ax.set_xlabel("Hour", size=14)
ax.set_ylabel(r'HGV ratio', size=14)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
#ax.set_ylim([0, 60])
#plt.title('Validation data v. forecast ')
plt.legend(prop={"size":14});
plt.show()
plt.show()
################################################################################

################################################################################
# ---------------------Diurnal Profile numbers of HGV and non-HGV --------------
f, ax = plt.subplots(2,1,figsize=(15, 10))
sns.boxplot(data=frame_traff,x=frame_traff.index.hour, y=frame_traff['Class4Volume'],hue='lockdown', ax=ax[0]).set(xlabel='Hour of day',ylabel='HGV numbers')
sns.boxplot(data=frame_traff,x=frame_traff.index.hour, y=frame_traff['non HGV'],hue='lockdown', ax=ax[1]).set(xlabel='Hour of day',ylabel='non HGV numbers')
ax[1].set_xlabel("Hour", size=14)
ax[1].set_ylabel(r'non HGV volume', size=14)
ax[0].set_xlabel("Hour", size=14)
ax[0].set_ylabel(r'HGV volume', size=14)
ax[1].tick_params(axis="x", labelsize=14)
ax[1].tick_params(axis="y", labelsize=14)
ax[0].tick_params(axis="x", labelsize=14)
ax[0].tick_params(axis="y", labelsize=14)
plt.show()
################################################################################
