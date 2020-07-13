##########################################################################################
#    Script to fit and predict using the Facebook time series model                      #
#    and traffic data                                                                    #
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

# File to download AURN air quality data from uk-air.defra.gov.uk
# Converts R files into Pandas dataframes

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
import datetime
from pathlib import Path
from windrose import WindroseAxes

# In this file we first pull all of the data from the DEFRA portal before fitting
# prophet models to each site. This is currently the same as a seperate Script
# for downloaded and analysing the DEFRA data here:
# https://github.com/loftytopping/DEFRA_Air_Quality_data
# The plan is to make this generic for any air quality data

#################################################################################
# Download required air quality future_data
# First check to see if data directory exists. If not, create it
download_path = "/AURN_data_download"
Path(download_path).mkdir(parents=True, exist_ok=True)

prophet_analysis_path = "/AURN_prophet_analysis"
Path(prophet_analysis_path).mkdir(parents=True, exist_ok=True)

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

# In the following we now download the data. Here we have a number of options
# - Specify the years to download data for
# - Specify the local authority[ies] to download data for
# - Download data for all authorities

#pdb.set_trace()

# Downloading site data for a specific year or years
years = [2016,2017,2018,2019,2020]
current_year = datetime.datetime.now().year

# If a single year is passed then convert to a list with a single value
if type(years) is int:
    years = [years]
datetime.datetime.now().year
# List authorities manually, or fit to all?
manual_selection = True
save_to_csv = False
# dont_update allows us to not automatically download new AURN data
dont_update = True
site_data_dict=dict()
site_data_dict_name=dict()

if manual_selection is True:
    list_authorities = ['Manchester']
else:
    list_authorities = metadata['AURN_metadata'].local_authority.unique()

for local_authority in list_authorities:

    # Does the authority data exist?
    #Path("/AURN_data_download/"+local_authority+"/").mkdir(parents=True, exist_ok=True)
    data_path = "/AURN_data_download/"+local_authority+"/"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    #pdb.set_trace()
    subset_df = metadata['AURN_metadata'][metadata['AURN_metadata'].local_authority == local_authority]

    # Check to see if your requested years will work and if not, change it
    # to do this lets create two new columns of datetimes for earliest and latest
    datetime_start=pd.to_datetime(subset_df['start_date'].values, format='%Y/%m/%d').year
    #Problem with the end date is it could be ongoing. In which case, convert that entry into a date and to_datetime
    now = datetime.datetime.now()
    datetime_end_temp=subset_df['end_date'].values
    step=0
    for i in datetime_end_temp:
        if i == 'ongoing':
            datetime_end_temp[step]=str(now.year)+'-'+str(now.month)+'-'+str(now.day)
        step+=1
    datetime_end = pd.to_datetime(datetime_end_temp).year

    earliest_year = np.min(datetime_start)
    latest_year = np.max(datetime_end)

    ## Check year range now requested
    #if np.min(years) < earliest_year:
    #    print("Invalid start year. The earliest you can select for ", local_authority ," is ", earliest_year)
    #    sys.exit()
    #if np.max(years) > latest_year:
    #    print("Invalid end year. The latest you can select for ", local_authority ," is ", latest_year)
    #    sys.exit()

    # Create dictionary of all site data from entire download session
    clean_site_data=True

    ################################################################################
    # Cycle through all sites in the local local_authority


    for site in subset_df['site_id'].unique():

        # Do we want to check if the site has 'all' of the pollutants?
        #if check_all is True:
        #    if all(elem in ['O3', 'NO2', 'NO', 'PM2.5', 'temp', 'ws', 'wd'] for elem in metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site]['parameter'].values)
        station_name = metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site]['site_name'].values[0]
        # Create a list of dataframes per each yearly downloaded_data
        # Concatenate these at the end
        downloaded_site_data = []

        for year in years:
            #pdb.set_trace()
            try:
                download_url = "https://uk-air.defra.gov.uk/openair/R_data/"+site+"_"+str(year)+".RData"
                downloaded_file=site+"_"+str(year)+".RData"

                # Check to see if file exists or not. Special case for current year as updates on hourley basis
                filename_path = "/AURN_data_download/"+local_authority+"/"+downloaded_file
                if dont_update is False:
                    if os.path.isfile(filename_path) is True and year != current_year:
                        print("Data file already exists, will use this")
                    else:
                        if os.path.isfile(filename_path) is True and year == current_year:
                        # Remove downloaded .Rdata file [make this optional]
                            os.remove(filename_path)
                        print("Downloading data file for ", station_name ," in ",str(year))
                    #pdb.set_trace()
                        if dont_update is False:
                            wget.download(download_url,out="/AURN_data_download/"+local_authority+"/")

                # Read the RData file into a Pandas dataframe
                downloaded_data = pyreadr.read_r(filename_path)
                # Save dataframe to an excel file and add coordinates as reference data into first row
                # Add date downloaded into second row
                downloaded_data[site+"_"+str(year)]['latitude']=metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site].latitude.values[0]
                downloaded_data[site+"_"+str(year)]['longitude']=metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site].longitude.values[0]

                # Append to dataframe list
                downloaded_site_data.append(downloaded_data[site+"_"+str(year)])

            except:
                print("Couldnt download data from ", year, " for ", station_name)

        if len(downloaded_site_data) == 0:
            print("No data could be downloaded for ", station_name)
        #final_dataframe = pd.DataFrame()
        else:
            final_dataframe = pd.concat(downloaded_site_data, axis=0, ignore_index=True)

            final_dataframe['datetime'] = pd.to_datetime(final_dataframe['date'])
            final_dataframe =final_dataframe.sort_values(by='datetime',ascending=True)
            final_dataframe=final_dataframe.set_index('datetime')

            #Add a new column into the dataframe for Ox
            try:
                final_dataframe['Ox']=final_dataframe['NO2']+final_dataframe['O3']
            except:
                print("Could not create Ox entry for ", site)

            #Add a new column into the dataframe for Ox
            try:
                final_dataframe['NOx']=final_dataframe['NO2']+final_dataframe['NO']
            except:
                print("Could not create NOx entry for ", site)
            # Now save the data frame to a .csv file

            # Now clean the dataframe for missing entries - make this optional!!
            if clean_site_data is True:
            # It might be that not all sites record all pollutants here. In which case I think
            # we just need to cycle through each potential pollutant
                for entry in ['O3', 'NO2', 'NO', 'PM2.5', 'Ox', 'NOx','temp', 'ws', 'wd']:
                    if entry in final_dataframe.columns.values:
                        #pdb.set_trace()
                        final_dataframe=final_dataframe.dropna(subset=[entry])
            # Now save the dataframe as a .csv file
            if save_to_csv is True:
                final_dataframe.to_csv("/AURN_data_download/"+local_authority+"/"+site+'.csv', index = False, header=True)

            # Append entire dataframe to all site catalogue dictionary
            site_data_dict[site] = final_dataframe
            site_data_dict_name[site] = metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site]['site_name'].values[0]


#
################################################################################
# Now load the traffic data from TfGM_Drakewell

# These files give us an insight into representiveness of traffic data
# The following data is from the closes 'Journey Time' BLU measurement point to
# the AURN site.

frame_traff = pd.read_csv(r'C:\Users\Dave\Documents\Code\Git_repos\Prophet_forecasting_AQ\Traffic_Data\TfGM\pvr_2016-01-01_1597d_portland.csv')
# The additional option below is for a site on Portland street but away from Piccadilly Gardens
#frame_traff = pd.read_csv(r'C:\Users\Dave\Documents\Code\Developing\Traffic_analysis\TfGM_Drakewell\pvr_2016-01-01_1597d_1.csv')

# Extract data from Channel 1
frame_traff=frame_traff[frame_traff['LaneDescription']=='Channel 1']
frame_traff['datetime'] = pd.to_datetime(frame_traff['Sdate'])
frame_traff = frame_traff.sort_values(by='datetime',ascending=True)
frame_traff=frame_traff.set_index('datetime')


# Now merge the two dataframes, both Air Quality and Traffic, on the time index
combined_df=pd.merge(frame_traff,site_data_dict['MAN3'], left_index=True, right_index=True)
#remove duplicate entries in the index (downloaded multiple CSV files with overlapping times)
combined_df = combined_df.loc[~combined_df.index.duplicated(keep='first')]


#Now produce a box-plot for all entries in dataset.
combined_df["NO2"] = pd.to_numeric(combined_df["NO2"])
combined_df["O3"] = pd.to_numeric(combined_df["O3"])
combined_df["Volume"] = pd.to_numeric(combined_df["Volume"])
combined_df["NO2 per Volume"]=combined_df["NO2"]/combined_df["Volume"]
combined_df["log NO2 per Volume"]=np.log(combined_df["NO2"]/combined_df["Volume"])

f, ax = plt.subplots(2,1,figsize=(12, 5))
ax[1].plot(combined_df.index, combined_df['log NO2 per Volume'], color='r', label='actual')
#ax = f.gca()
ax[1].set(xlabel='Date', ylabel="Log(NO2/Traffic Volume)")
ax[1].set_xlim([datetime.date(2019, 9, 28), datetime.date(2020, 5, 10)])
ax[0].plot(combined_df.index, combined_df['Volume'], color='r', label='actual')
ax[0].set_xlim([datetime.date(2019, 9, 28), datetime.date(2020, 5, 10)])
ax[0].set(xlabel='Date', ylabel="Traffic Volume [per hour]")
plt.show()
plt.close('all')

pdb.set_trace()


# Mask to look at typical diurnal profiles before COVID19 lockdown
mask_pre_lockdown = (combined_df.index < '2020-3-23')
f, ax = plt.subplots(3,1,figsize=(12, 5))
sns.boxplot(x=combined_df.loc[mask_pre_lockdown].index.hour, y=combined_df.loc[mask_pre_lockdown]['NO2'],ax=ax[0])
sns.boxplot(x=combined_df.loc[mask_pre_lockdown].index.hour, y=combined_df.loc[mask_pre_lockdown]['O3'],ax=ax[1])
sns.boxplot(x=combined_df.loc[mask_pre_lockdown].index.hour, y=combined_df.loc[mask_pre_lockdown]['log NO2 per Volume'],ax=ax[2])
plt.show()
plt.close('all')
pdb.set_trace()

# Now produce a set of two combined box-plots according to March before 2020 and for 2020
mask_tag = (combined_df.index.month == 4) & (combined_df.index > '2019-12-30')
combined_df['April_2020'] = mask_tag
booleanDictionary = {True: 'TRUE', False: 'FALSE'}
f, ax = plt.subplots(4,1,figsize=(15, 15))
mask3 = (combined_df.index.month == 4)
sns.boxplot(data=combined_df.loc[mask3],x=combined_df.loc[mask3].index.hour, y=combined_df.loc[mask3]['NO2'],hue='April_2020', ax=ax[0])
sns.boxplot(data=combined_df.loc[mask3],x=combined_df.loc[mask3].index.hour, y=combined_df.loc[mask3]['O3'],hue='April_2020',ax=ax[1])
sns.boxplot(data=combined_df.loc[mask3],x=combined_df.loc[mask3].index.hour, y=combined_df.loc[mask3]['Volume'],hue='April_2020',ax=ax[2])
sns.boxplot(data=combined_df.loc[mask3],x=combined_df.loc[mask3].index.hour, y=combined_df.loc[mask3]['log NO2 per Volume'],hue='April_2020',ax=ax[3])
ax[0].tick_params(axis='x', which='both', bottom='off',labelbottom='off')
ax[1].tick_params(axis='x', which='both', bottom='off',labelbottom='off')
ax[2].tick_params(axis='x', which='both', bottom='off',labelbottom='off')
plt.show()
plt.close('all')
pdb.set_trace()


######################  Train a Prophet instance to the NO2 per volume ###########################
train_dataset2= pd.DataFrame()
train_dataset2['ds'] = (pd.to_datetime(combined_df['Sdate']))
train_dataset2['O3']=combined_df['O3']
train_dataset2['y']=combined_df['log NO2 per Volume']
#train_dataset2['y']=combined_df['NO2 per Volume']
train_dataset2['Modelled Wind Direction']=combined_df['wd']
train_dataset2['Modelled Wind Speed']=combined_df['ws']
train_dataset2['Modelled Temperature']=combined_df['temp']
train_dataset2['Traffic Volume']=combined_df['Volume']
train_dataset2['NO2']=combined_df['NO2']
train_dataset2 = train_dataset2[train_dataset2.ds != 'End']
train_dataset2 = train_dataset2[train_dataset2['O3'] != 'No data']
train_dataset2 = train_dataset2[train_dataset2['y'] != 'No data']
train_dataset2 = train_dataset2[train_dataset2['Modelled Wind Direction'] != 'No data']
train_dataset2 = train_dataset2[train_dataset2['Modelled Wind Speed'] != 'No data']
train_dataset2 = train_dataset2[train_dataset2['Modelled Temperature'] != 'No data']
train_dataset2=train_dataset2.replace([np.inf, -np.inf], np.nan)
train_dataset2.dropna(inplace=True)
pro_regressor2= Prophet()
# Add additional regressors
pro_regressor2.add_regressor('Modelled Wind Direction')
pro_regressor2.add_regressor('Modelled Wind Speed')
pro_regressor2.add_regressor('Modelled Temperature')
#pro_regressor2.add_regressor('Traffic Volume')
mask_reg1b = (train_dataset2.ds < '2020-3-01')
mask_reg2b = (train_dataset2.ds >= '2020-3-01')
mask_reg3b = (train_dataset2.ds >= '2020-3-25')
mask_futureb = (train_dataset2.ds > '2019-12-01')
# Specify a train and test dataset. Train before March 2020
train_X2= train_dataset2.loc[mask_reg1b]
test_X2= train_dataset2.loc[mask_reg2b]

pro_regressor2.fit(train_X2)
forecast_data2 = pro_regressor2.predict(test_X2)
# From the forecast log(No2/Volume), use the traffic data to predict the levels of NO2
forecast_data2['NO2 from volume']=np.exp(forecast_data2['yhat'].values)*test_X2['Traffic Volume'].values
forecast_data2['NO2 from volume upper']=np.exp(forecast_data2['yhat_upper'].values)*test_X2['Traffic Volume'].values
forecast_data2['NO2 from volume lower']=np.exp(forecast_data2['yhat_lower'].values)*test_X2['Traffic Volume'].values


# Now plot the 'business as usual' NO2 per volume versus the actual calculated from combining
# the AURN data and traffic data
fig =pro_regressor2.plot(forecast_data2, uncertainty=True,figsize=(15, 5))
axes = fig.get_axes()
plt.plot(train_dataset2.loc[mask_reg2b]['ds'], train_dataset2.loc[mask_reg2b]['y'], color='r', label='actual')
ax = fig.gca()
ax.set_xlim([datetime.date(2020, 2, 28), datetime.date(2020, 5, 10)])
ax.set_ylim([-4, 1.5])
axes[0].set_xlabel('Date')
axes[0].set_ylabel('log (NO2 / Traffic volume)')
plt.title('Validation data v. forecast - log NO2 / Traffic volume')
plt.legend();
plt.show()
plt.close('all')
pdb.set_trace()
#########################################################################################

###########  Train to the NO2, as per our standard Propher use  #########################
# Now fit two prophet models, one to the just met data and one to the traffic
train_dataset= pd.DataFrame()
train_dataset['ds'] = (pd.to_datetime(combined_df['Sdate']))
train_dataset['O3']=combined_df['O3']
train_dataset['y']=combined_df['NO2']
train_dataset['Modelled Wind Direction']=combined_df['wd']
train_dataset['Modelled Wind Speed']=combined_df['ws']
train_dataset['Modelled Temperature']=combined_df['temp']
train_dataset['Traffic Volume']=combined_df['Volume']
train_dataset['NO2 per Volume']=combined_df['NO2 per Volume']
train_dataset = train_dataset[train_dataset.ds != 'End']
train_dataset = train_dataset[train_dataset['O3'] != 'No data']
train_dataset = train_dataset[train_dataset['y'] != 'No data']
train_dataset = train_dataset[train_dataset['Modelled Wind Direction'] != 'No data']
train_dataset = train_dataset[train_dataset['Modelled Wind Speed'] != 'No data']
train_dataset = train_dataset[train_dataset['Modelled Temperature'] != 'No data']
train_dataset=train_dataset.replace([np.inf, -np.inf], np.nan)
train_dataset.dropna(inplace=True)
mask_reg1 = (train_dataset.ds < '2020-3-01')
mask_reg2 = (train_dataset.ds >= '2020-3-01')
mask_reg3 = (train_dataset.ds >= '2020-3-25')
mask_future = (train_dataset.ds > '2019-12-01')

# Build a regressor [using a changpoint scale inferred from the Cross Validation studies]
pro_regressor= Prophet(changepoint_prior_scale=10)
#pro_regressor.add_country_holidays(country_name='UK')
pro_regressor.add_regressor('Modelled Wind Direction')
pro_regressor.add_regressor('Modelled Wind Speed')
pro_regressor.add_regressor('Modelled Temperature')
train_X= train_dataset.loc[mask_reg1]
test_X= train_dataset.loc[mask_reg2]

pro_regressor.fit(train_X)
forecast_data = pro_regressor.predict(test_X)

# Plot the actual data with forecast NO2 AND NO2 from log(NO2/volume) fits
fig =pro_regressor.plot(forecast_data, uncertainty=True,figsize=(15, 5), xlabel='Date', ylabel=r'NO2 $\mu g.m^{-3}$')
plt.plot(train_dataset.loc[mask_reg2]['ds'], train_dataset.loc[mask_reg2]['y'], color='r', label='Measured')
plt.plot(forecast_data['ds'], forecast_data['yhat'], color='tab:blue', label='Forecast')
plt.plot(forecast_data2['ds'], forecast_data2['NO2 from volume'], color='g', label='Forecast using traffic data')
#plt.plot(forecast_data2['ds'], forecast_data2['NO2 from volume upper'], color='g',linestyle='--')
#plt.plot(forecast_data2['ds'], forecast_data2['NO2 from volume lower'], color='g',linestyle='--')
#plt.fill_between(forecast_data2['ds'], forecast_data2['NO2 from volume lower'], forecast_data2['NO2 from volume upper'], color='green', alpha=0.1)
ax = fig.gca()
ax.set_xlim([datetime.date(2020, 2, 28), datetime.date(2020, 5, 10)])
ax.set_ylim([0, 120])
ax.set_xlabel("Date", size=14)
ax.set_ylabel(r'NO2 $\mu g.m^{-3}$', size=14)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
#plt.title('Validation data v. forecast ')
plt.legend(prop={"size":14});
plt.show()
plt.close('all')

# Calculate the % deviation from the predictions based on traffic
forecast_data2['% deviation']=(forecast_data2['NO2 from volume'].values-forecast_data['yhat'].values)/(forecast_data['yhat'].values)*100.0
mask_reg4 = (forecast_data2.ds >= '2020-3-25')

# Create a windrose plot showing where largest deviations occur
#fig = plt.figure(figsize=(15, 10), projection='windrose')
ax = WindroseAxes.from_ax()
ax.bar(train_dataset2.loc[mask_reg3b]['Modelled Wind Direction'].values, forecast_data2.loc[mask_reg4]['% deviation'].values)
ax.set_legend()
ax.legend(title='% Deviation',loc='upper left')
plt.show()

pdb.set_trace()

# Create a boxplot looking at measured, forecast with and without traffic
# To do this we are going to concatenate vertically
forecast_normal_df = forecast_data[['ds','yhat']]
forecast_normal_new_df = forecast_normal_df.copy()
forecast_normal_new_df['label'] = 'Forecast'
forecast_normal_new_df=forecast_normal_new_df.rename(columns={"yhat": "y"})
forecast_traffic_df = forecast_data2[['ds','NO2 from volume']]
forecast_traffic_new_df = forecast_traffic_df.copy()
forecast_traffic_new_df['label'] = 'Forecast using traffic data'
forecast_traffic_new_df=forecast_traffic_new_df.rename(columns={"NO2 from volume": "y"})
measured_df = train_dataset.loc[mask_reg2][['ds','y']]
measured_new_df = measured_df.copy()
measured_new_df['label'] = 'Measured'
vertical_stack = pd.concat([forecast_normal_new_df, forecast_traffic_new_df], axis=0)
vertical_stack = pd.concat([vertical_stack, measured_new_df], axis=0)
vertical_stack['ds']=pd.to_datetime(vertical_stack['ds'])
vertical_stack=vertical_stack.set_index('ds')
# Now set the index to be the datetime
f, ax = plt.subplots(1,1,figsize=(12, 5))
sns.boxplot(data=vertical_stack,x=vertical_stack.index.hour, y=vertical_stack['y'],hue='label')
ax.set_xlabel("Hour", size=14)
ax.set_ylabel(r'NO2 $\mu g.m^{-3}$', size=14)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
#plt.title('Validation data v. forecast ')
plt.legend(prop={"size":14});
plt.show()
