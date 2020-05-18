##########################################################################################
#    Script to assess performance of time series model according to site type            #
#    Requires us to load data already generated through the cross validation of          #
#    the prophet fit and forecasts in the file 'automated_prophet_all_site_analyses.py'  #
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
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from scipy import stats

# This script works on data already generated through the cross validation of
# the prophet fit and forecasts

# The idea is to asses performance in terms of total concentrations by site type
# then also by diurnal trends

# We retain the same directory structures as the original scripts

#################################################################################
# Where did you save your output from the Cross Validation fits?
prophet_analysis_path = "/AURN_prophet_analysis"
#Path(prophet_analysis_path).mkdir(parents=True, exist_ok=True)

meta_data_url = "https://uk-air.defra.gov.uk/openair/R_data/AURN_metadata.RData"
data_url = "https://uk-air.defra.gov.uk/openair/R_data/"
# Do you want to check the sites listed in the meta data file?
# Does the metadatafile exist?
meata_data_filename = 'AURN_metadata.RData'
if os.path.isfile(meata_data_filename) is True:
    print("Meta data file already exists in this directory, will use this")
else:
    print("Downloading Meta data file")
    wget.download(meta_data_url)

# Read the RData file into a Pandas dataframe
metadata = pyreadr.read_r(meata_data_filename)

# We do not download any AURN data in this script, since this will have taken
# place during the Prophet fitting process
# Nonetheless we can still choose to look at individual authorities

manual_selection = False
save_to_csv = False
site_data_dict=dict()
site_data_dict_name=dict()

if manual_selection is True:
    list_authorities = ['Manchester']
else:
    list_authorities = metadata['AURN_metadata'].local_authority.unique()

# Initialise a set of lists to store model performance stats
mean_deviation=[]
site_type_list=[]
name_list=[]
station_count=1
mean_percen=[]
outside_error_list=[]
R_list=[]
inside_uncertainty=[]

for local_authority in list_authorities:


    skip_authority=False
    # Does the authority data exist?
    #Path("/AURN_data_download/"+local_authority+"/").mkdir(parents=True, exist_ok=True)
    data_path = "C:/Users/Dave/Documents/Code/AURN_prophet_analysis/"+local_authority+"/"
    if not os.path.exists(data_path):
        print("Directory does not exist for ", local_authority, "so we will skip this")
        skip_authority=True

    if skip_authority is False:

        #pdb.set_trace()
        subset_df = metadata['AURN_metadata'][metadata['AURN_metadata'].local_authority == local_authority]

        for site in subset_df['site_id'].unique():

            station_name = metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site]['site_name'].values[0]
            site_type = metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site].location_type.values[0]
            textstr = local_authority+', '+station_name+', '+site_type

            try:
                station_name = metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site]['site_name'].values[0]
                name_list.append(station_name)
                site_forecast_data = pd.read_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_30days.csv")
                site_forecast_data['datetime'] = pd.to_datetime(site_forecast_data['ds'])
                site_forecast_data[station_name] = (site_forecast_data['yhat']-site_forecast_data['y'])

                max_x = max(site_forecast_data.y)

                site_forecast_data_percen = pd.read_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_30days.csv")
                site_forecast_data_percen['datetime'] = pd.to_datetime(site_forecast_data['ds'])
                site_forecast_data_percen[station_name] = ((site_forecast_data['yhat']-site_forecast_data['y'])/site_forecast_data['y'])*100.0

                # Calculate statistical correlation. Use Pearson
                R,p=stats.pearsonr(site_forecast_data['yhat'].values, site_forecast_data['y'].values)
                if p <= 0.05:
                    R_list.append(R)

                mask_in_certainty = (site_forecast_data['y'] > site_forecast_data['yhat_lower']) & (site_forecast_data['y'] < site_forecast_data['yhat_upper'])
                site_forecast_data['in bounds']=mask_in_certainty
                inside_uncertainty.append((site_forecast_data['in bounds'].sum()/len(site_forecast_data['in bounds'].values))*100)

                if station_count == 1:
                    collection_df = site_forecast_data[['datetime', station_name]].copy()
                    collection_df_percen = site_forecast_data_percen[['datetime', station_name]].copy()
                    station_count+=1
                else:
                    collection_df[station_name]=site_forecast_data[station_name]
                    collection_df_percen[station_name]=site_forecast_data_percen[station_name]

                #outside_error_list.append(outside_error)
                mean_deviation.append(site_forecast_data[station_name].mean())
                mean_percen.append(site_forecast_data_percen[station_name].mean())
                site_type_list.append(metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site].location_type.values[0])

            except:
                print("Problem with ", station_name)

# Define a set of colours for visualising performance by AURN site type [Urban Background, Rural etc..]
colour_list=['tab:green','tab:red','tab:cyan','tab:grey','tab:orange','tab:purple']
sns.set(font_scale=1.5)

################################################################################
# Now turn the mean deviation data into a dataframe
data_tuple = list(zip(mean_deviation,site_type_list))
mean_data = pd.DataFrame(data_tuple, columns=['Mean Deviation','site type'])
# Plot the mean deviation as a function of site type
fig = plt.figure(figsize=(20, 10))
colour_step=0
for entry in mean_data['site type'].unique():
    df = mean_data[mean_data['site type'] == entry]
    sns.distplot(df['Mean Deviation'], hist = True, kde = True, label=entry,color=colour_list[colour_step])
    colour_step+=1
# Plot formatting
plt.legend(prop={'size': 12})
plt.title('Mean Deviation for site types')
plt.xlabel('Mean Deviation')
plt.ylabel('Normalised density')
plt.show()
pdb.set_trace()
plt.close('all')
################################################################################

################################################################################
# Now turn the mean % deviation data into a dataframe
data_tuple2 = list(zip(mean_percen,site_type_list))
mean_data_percen = pd.DataFrame(data_tuple2, columns=['% Deviation','site type'])
mean_data_percen=mean_data_percen.replace([np.inf, -np.inf], np.nan)
mean_data_percen=mean_data_percen.dropna()
# Plot the mean deviation as a function of site type
fig = plt.figure(figsize=(20, 10))
colour_step=0
for entry in mean_data_percen['site type'].unique():
    df = mean_data_percen[mean_data_percen['site type'] == entry]
    sns.distplot(df['% Deviation'], hist = True, kde = True, label=entry,color=colour_list[colour_step])
    colour_step+=1
# Plot formatting
plt.legend(prop={'size': 12})
plt.title('Percentage Deviation for site types')
plt.xlabel('% Deviation ')
plt.ylabel('Normalised density')
plt.show()
pdb.set_trace()
plt.close('all')
################################################################################

################################################################################
# Now turn the mean deviation data into a dataframe, but restrict the range on
# the x axis for a 'zoom' in on the majority of sites
data_tuple2 = list(zip(mean_percen,site_type_list))
mean_data_percen = pd.DataFrame(data_tuple2, columns=['% Deviation','site type'])
mean_data_percen=mean_data_percen.replace([np.inf, -np.inf], np.nan)
mean_data_percen=mean_data_percen.dropna()
# Plot the mean deviation as a function of site type
fig = plt.figure(figsize=(20, 10))
colour_step=0
for entry in mean_data_percen['site type'].unique():
    df = mean_data_percen[mean_data_percen['site type'] == entry]
    sns.distplot(df['% Deviation'], hist = True, kde = True, label=entry,color=colour_list[colour_step])
    colour_step+=1
# Plot formatting
plt.legend(prop={'size': 12})
plt.title('Percentage Deviation for site types')
plt.xlabel('% Deviation ')
plt.ylabel('Normalised density')
plt.xlim(-50, 120)
plt.show()
pdb.set_trace()
plt.close('all')
################################################################################

################################################################################
# Now turn the Pearsons R data into a dataframe
data_tuple4 = list(zip(R_list,site_type_list))
R_list_data = pd.DataFrame(data_tuple4, columns=['Corr','site type'])
# Plot the distribution of R as a function of site type
fig = plt.figure(figsize=(20, 10))
colour_step=0
for entry in R_list_data['site type'].unique():
    df = R_list_data[R_list_data['site type'] == entry]
    sns.distplot(df['Corr'], hist = True, kde = True, label=entry,color=colour_list[colour_step])
    colour_step+=1
# Plot formatting
plt.legend(prop={'size': 12})
plt.title('Pearson R for site types')
plt.xlabel('Pearson R')
plt.ylabel('Normalised density')
plt.show()
pdb.set_trace()
plt.close('all')
################################################################################

################################################################################
# Now store the frequency of observations within forecast uncertainty
data_tuple5 = list(zip(inside_uncertainty,site_type_list))
inbounds_list_data = pd.DataFrame(data_tuple5, columns=['%','site type'])
# Plot the mean deviation as a function of site type
fig = plt.figure(figsize=(20, 10))
colour_step=0
for entry in inbounds_list_data['site type'].unique():
    df = inbounds_list_data[inbounds_list_data['site type'] == entry]
    sns.distplot(df['%'], hist = True, kde = True, label=entry,color=colour_list[colour_step])
    colour_step+=1
# Plot formatting
plt.legend(prop={'size': 12})
plt.title('Percentage of observations within confidence interval site types')
plt.xlabel('Percentage')
plt.ylabel('Normalised density')
plt.show()
pdb.set_trace()
plt.close('all')
################################################################################
