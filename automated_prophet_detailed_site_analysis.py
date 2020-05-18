##########################################################################################
#    Script to fit and predict using the Facebook time series model                      #
#    Requires us to download AURN air quality data from uk-air.defra.gov.uk              #
#    Also includes ability to plot diurnal profile comparisons                           #
#    This script performs 'detailed' hyperparameter sensitivity investigations           #
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

# In this file we first pull all of the data from the DEFRA portal before fitting
# prophet models to each site. This is currently the same as a seperate Script
# for downloaded and analysing the DEFRA data here:
# https://github.com/loftytopping/DEFRA_Air_Quality_data
# The plan is to make this generic for any air quality data

# Please note, as stated above, this file is intended to focus on performing a
# range of model parameter sensitivity tests so focuses on one authority. This
# can be changed according to user needs but I have not tested for scaling yet

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

# Downloading site data for a specific year or years
years = [2015,2016,2017,2018,2019,2020]
# If a single year is passed then convert to a list with a single value
if type(years) is int:
    years = [years]
current_year = datetime.datetime.now()
# List authorities manually, or fit to all?
manual_selection = True
save_to_csv = False
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
                if os.path.isfile(filename_path) is True and year != current_year:
                    print("Data file already exists, will use this")
                else:
                    if os.path.isfile(filename_path) is True and year == current_year:
                        # Remove downloaded .Rdata file [make this optional]
                        os.remove(filename_path)
                    print("Downloading data file for ", station_name ," in ",str(year))
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



for site in site_data_dict.keys():

    # Now fit and predict from the Prophet model. We are going to use cross validation on each site for the 2019 dataset
    # We are going to produce rolling forecasts for 1 month concentrations. For the first set of simulations we are going
    # to use 3 years worth of data, but then see how numbers change when we use two years

    # The plan is to then visualise each site on a boxplot by extracting the entire MAPE values from Prophet
    # Now fit the timeseries model to the dataset, looking at NO2

    # For the cross validation we want to keep the data stopping at the end of 2019 so lets add a mask for our DataFrame
    # We thus need to

    station_name = metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site]['site_name'].values[0]

    prophet_data_path = prophet_analysis_path+"/"+local_authority+"/"
    if not os.path.exists(prophet_data_path):
        os.makedirs(prophet_data_path)


    try:

        mask_cv = site_data_dict[site].index <= '2019-12-31'

        train_dataset= pd.DataFrame()
        train_dataset['ds'] = pd.to_datetime(site_data_dict[site].loc[mask_cv]['date'])
        #train_dataset['Ozone']=frame['Ozone']
        train_dataset['y']=pd.to_numeric(site_data_dict[site].loc[mask_cv]['NO2'])
        train_dataset['Modelled Wind Direction']=pd.to_numeric(site_data_dict[site].loc[mask_cv]['wd'])
        train_dataset['Modelled Wind Speed']=pd.to_numeric(site_data_dict[site].loc[mask_cv]['ws'])
        train_dataset['Modelled Temperature']=pd.to_numeric(site_data_dict[site].loc[mask_cv]['temp'])
        #pdb.set_trace()
        train_dataset=train_dataset.dropna()
        # Build a regressor
        pro_regressor=Prophet()
        pro_regressor.add_regressor('Modelled Wind Direction')
        pro_regressor.add_regressor('Modelled Wind Speed')
        pro_regressor.add_regressor('Modelled Temperature')
        pro_regressor.fit(train_dataset)

        # 30 day horizon
        df_cv_30=cross_validation(model=pro_regressor, initial='1095 days', horizon='30 days')
        df_cv_30.to_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_30days.csv", index = False, header=True)
        df_p_30 = performance_metrics(df_cv_30)
        # Save a figure to file
        #check if directory exists
        fig = plot_cross_validation_metric(df_cv_30, metric='mape')
        plt.savefig(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-MAPE_30days.png")
        plt.close('all')
        df_p_30.to_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-MAPE_30days.csv", index = False, header=True)
        # 60 day horizon
        df_cv_60=cross_validation(model=pro_regressor, initial='1095 days', horizon='60 days')
        df_cv_60.to_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_60days.csv", index = False, header=True)
        df_p_60 = performance_metrics(df_cv_60)
        # Save a figure to file
        #check if directory exists
        fig = plot_cross_validation_metric(df_cv_60, metric='mape')
        plt.savefig(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-MAPE_60days.png")
        plt.close('all')
        df_p_60.to_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-MAPE_60days.csv", index = False, header=True)
        #pdb.set_trace()

        # Now do the 30 days but using 2 years worth of data instead
        df_cv_30_2years=cross_validation(model=pro_regressor, initial='730 days', horizon='30 days')
        df_cv_30_2years.to_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_30days_2years.csv", index = False, header=True)
        df_p_30_2years = performance_metrics(df_cv_30_2years)
        # Save a figure to file
        #check if directory exists
        fig = plot_cross_validation_metric(df_cv_30_2years, metric='mape')
        plt.savefig(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-MAPE_30days_2years.png")
        plt.close('all')
        df_p_30_2years.to_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-MAPE_30days_2years.csv", index = False, header=True)

        # Now do 6 months!
        df_cv_30_6months=cross_validation(model=pro_regressor, initial='180 days', horizon='30 days')
        df_cv_30_6months.to_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_30days_6months.csv", index = False, header=True)
        df_p_30_6months = performance_metrics(df_cv_30_6months)
        # Save a figure to file
        #check if directory exists
        fig = plot_cross_validation_metric(df_cv_30_6months, metric='mape')
        plt.savefig(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-MAPE_30days_6months.png")
        plt.close('all')
        df_p_30_6months.to_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-MAPE_30days_6months.csv", index = False, header=True)

        #Now change the number of changepoints to 1 per month
        pro_regressor_cp=Prophet(changepoint_prior_scale=10)
        pro_regressor_cp.add_regressor('Modelled Wind Direction')
        pro_regressor_cp.add_regressor('Modelled Wind Speed')
        pro_regressor_cp.add_regressor('Modelled Temperature')
        pro_regressor_cp.fit(train_dataset)
        df_cv_30_cp=cross_validation(model=pro_regressor_cp, initial='1095 days', horizon='30 days')
        df_cv_30_cp.to_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_30days_cpcsv", index = False, header=True)
        df_p_30_cp = performance_metrics(df_cv_30_cp)
        # Save a figure to file
        #check if directory exists
        fig = plot_cross_validation_metric(df_cv_30_cp, metric='mape')
        plt.savefig(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-MAPE_30days_cp.png")
        plt.close('all')
        df_p_30_cp.to_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-MAPE_30days_cp.csv", index = False, header=True)
        # Now extract the validation metrics into a seperate DataFrame
        #validation_dataframe = pd.DataFrame()
        #validation_dataframe['1month_forecast_error']
    except:
        print("Error in generating all CV stats")
