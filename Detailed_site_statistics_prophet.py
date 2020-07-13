##########################################################################################
#    Script to assess performance of time series model according to site type            #
#    Requires us to download AURN air quality data from uk-air.defra.gov.uk              #
#    Also includes ability to plot diurnal profile comparisons                           #
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

# Please note this file was written to look at the various model varients as applied to
# the Manchester Piccadilly site from our original study. If looking at another site, you
# will need to change some 'hard coded' links here. This file is run after you run:
# automated_prophet_detailed_site_analysis.py in the same directory

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
import matplotlib.ticker as ticker
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from scipy import stats
from sklearn.preprocessing import PowerTransformer
import scipy

# This script works on data already generated through the cross validation of
# the prophet fit and forecasts

# The idea is to asses performance in terms of total concentrations by site type
# then also by diurnal trends

# We retain the same directory structures as the original scripts

#################################################################################
# 1) Download required air quality future_data
# First check to see if data directory exists. If not, create it
#download_path = "/AURN_data_download"
#Path(download_path).mkdir(parents=True, exist_ok=True)

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

# In the following we load in the data. Here we have a number of options
# - Specify the years to download data for
# - Specify the local authority[ies] to download data for
# - Download data for all authorities

# Downloading site data for a specific year or years
#years = [2015,2016,2017,2018,2019,2020]
# If a single year is passed then convert to a list with a single value
#if type(years) is int:
#    years = [years]
#current_year = datetime.datetime.now()
# List authorities manually, or fit to all?
manual_selection = True
save_to_csv = False
site_data_dict=dict()
site_data_dict_name=dict()

if manual_selection is True:
    list_authorities = ['Manchester']
else:
    list_authorities = metadata['AURN_metadata'].local_authority.unique()


mean_deviation=[]
site_type_list=[]
name_list=[]
station_count=1
mean_percen=[]
outside_error_list=[]
R_list=[]

for local_authority in list_authorities:


    skip_authority=False
    # Does the authority data exist?
    #Path("/AURN_data_download/"+local_authority+"/").mkdir(parents=True, exist_ok=True)
    data_path = "/AURN_prophet_analysis/"+local_authority+"/"
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
                #pdb.set_trace()
                site_forecast_data=site_forecast_data.rename(columns={station_name: "Prophet1"})

                site_forecast_data_percen = pd.read_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_30days.csv")
                site_forecast_data_percen['datetime'] = pd.to_datetime(site_forecast_data['ds'])
                site_forecast_data_percen[station_name] = ((site_forecast_data['yhat']-site_forecast_data['y'])/site_forecast_data['y'])*100.0
                site_forecast_data_percen=site_forecast_data_percen.rename(columns={station_name: "Prophet1"})
                #pdb.set_trace()
                #R,p=stats.pearsonr(site_forecast_data['yhat'].values, site_forecast_data['y'].values)
                #if p <= 0.05:
                #    R_list.append(R)

                site_forecast_data_2years = pd.read_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_30days_2years.csv")
                site_forecast_data_2years['datetime'] = pd.to_datetime(site_forecast_data_2years['ds'])
                site_forecast_data_2years[station_name] = (site_forecast_data_2years['yhat']-site_forecast_data_2years['y'])

                site_forecast_data_percen_2years = pd.read_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_30days_2years.csv")
                site_forecast_data_percen_2years['datetime'] = pd.to_datetime(site_forecast_data_2years['ds'])
                site_forecast_data_percen_2years[station_name] = ((site_forecast_data_2years['yhat']-site_forecast_data_2years['y'])/site_forecast_data_2years['y'])*100.0
                site_forecast_data_percen_2years=site_forecast_data_percen_2years.rename(columns={station_name: "Prophet2"})
                #pdb.set_trace()

                site_forecast_data_6months = pd.read_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_30days_6months.csv")
                site_forecast_data_6months['datetime'] = pd.to_datetime(site_forecast_data_6months['ds'])
                site_forecast_data_6months[station_name] = (site_forecast_data_6months['yhat']-site_forecast_data_6months['y'])

                site_forecast_data_percen_6months = pd.read_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_30days_6months.csv")
                site_forecast_data_percen_6months['datetime'] = pd.to_datetime(site_forecast_data_6months['ds'])
                site_forecast_data_percen_6months[station_name] = ((site_forecast_data_6months['yhat']-site_forecast_data_6months['y'])/site_forecast_data_6months['y'])*100.0
                site_forecast_data_percen_6months=site_forecast_data_percen_6months.rename(columns={station_name: "Prophet3"})
                #pdb.set_trace()

                site_forecast_data_cp = pd.read_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_30days_cp.csv")
                site_forecast_data_cp['datetime'] = pd.to_datetime(site_forecast_data_cp['ds'])
                site_forecast_data_cp[station_name] = (site_forecast_data_cp['yhat']-site_forecast_data_cp['y'])

                site_forecast_data_percen_cp = pd.read_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_30days_cp.csv")
                site_forecast_data_percen_cp['datetime'] = pd.to_datetime(site_forecast_data_cp['ds'])
                site_forecast_data_percen_cp[station_name] = ((site_forecast_data_cp['yhat']-site_forecast_data_cp['y'])/site_forecast_data_cp['y'])*100.0
                site_forecast_data_percen_cp=site_forecast_data_percen_cp.rename(columns={station_name: "Prophet4"})

                site_forecast_data_pt = pd.read_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_30days_pt.csv")
                site_forecast_data_pt['datetime'] = pd.to_datetime(site_forecast_data_pt['ds'])
                site_forecast_data_pt[station_name] = (site_forecast_data_pt['yhat_pt']-site_forecast_data_pt['y_pt'])

                site_forecast_data_percen_pt = pd.read_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_30days_pt.csv")
                site_forecast_data_percen_pt['datetime'] = pd.to_datetime(site_forecast_data_pt['ds'])
                site_forecast_data_percen_pt[station_name] = ((site_forecast_data_pt['yhat_pt']-site_forecast_data_pt['y_pt'])/site_forecast_data_pt['y_pt'])*100.0
                site_forecast_data_percen_pt=site_forecast_data_percen_pt.rename(columns={station_name: "Prophet5"})

                site_forecast_data_pt_cp = pd.read_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_30days_pt_cp.csv")
                site_forecast_data_pt_cp['datetime'] = pd.to_datetime(site_forecast_data_pt_cp['ds'])
                site_forecast_data_pt_cp[station_name] = (site_forecast_data_pt_cp['yhat_pt']-site_forecast_data_pt_cp['y_pt'])

                site_forecast_data_percen_pt_cp = pd.read_csv(prophet_analysis_path+"/"+local_authority+"/"+station_name+"-forecast_30days_pt_cp.csv")
                site_forecast_data_percen_pt_cp['datetime'] = pd.to_datetime(site_forecast_data_pt_cp['ds'])
                site_forecast_data_percen_pt_cp[station_name] = ((site_forecast_data_pt_cp['yhat_pt']-site_forecast_data_pt_cp['y_pt'])/site_forecast_data_pt_cp['y_pt'])*100.0
                site_forecast_data_percen_pt_cp=site_forecast_data_percen_pt_cp.rename(columns={station_name: "Prophet6"})


                #fig = plt.figure(figsize=(20, 10))
                #plt.text(0.02, 0.9, textstr, fontsize=12, transform=plt.gcf().transFigure)
                plot = sns.jointplot(x="yhat_pt", y="y_pt", data=site_forecast_data_pt, kind="hex").set_axis_labels("Predicted", "Measured")
                plot.ax_joint.plot(site_forecast_data_pt.y_pt,site_forecast_data_pt.y_pt, 'r-', linewidth = 1)
                plot.ax_joint.plot(site_forecast_data_pt.y_pt,site_forecast_data_pt.y_pt*1.25, 'b-.', linewidth = 0.8)
                plot.ax_joint.plot(site_forecast_data_pt.y_pt,site_forecast_data_pt.y_pt*0.75, 'b-.', linewidth = 0.8)
                min_x = min(site_forecast_data_pt.y_pt)
                max_x = max(site_forecast_data_pt.y_pt)
                plot.ax_joint.set_xlim(min_x,max_x)
                plot.ax_joint.set_ylim(min_x,max_x)
                #plot.fig.suptitle(textstr)
                plot.savefig(data_path+station_name+"_forecast_30days_pt_Hexjoint_Picc.png")
                plt.show()
                plt.close('all')

                #fig = plt.figure(figsize=(20, 10))
                #plt.text(0.02, 0.9, textstr, fontsize=12, transform=plt.gcf().transFigure)
                plot = sns.jointplot(x="yhat_pt", y="y_pt", data=site_forecast_data_pt_cp, kind="hex").set_axis_labels("Predicted", "Measured")
                plot.ax_joint.plot(site_forecast_data_pt.y_pt,site_forecast_data_pt.y_pt, 'r-', linewidth = 1)
                plot.ax_joint.plot(site_forecast_data_pt.y_pt,site_forecast_data_pt.y_pt*1.25, 'b-.', linewidth = 0.8)
                plot.ax_joint.plot(site_forecast_data_pt.y_pt,site_forecast_data_pt.y_pt*0.75, 'b-.', linewidth = 0.8)
                min_x = min(site_forecast_data_pt.y_pt)
                max_x = max(site_forecast_data_pt.y_pt)
                plot.ax_joint.set_xlim(min_x,max_x)
                plot.ax_joint.set_ylim(min_x,max_x)
                #plot.fig.suptitle(textstr)
                plot.savefig(data_path+station_name+"_forecast_30days_pt_cp_Hexjoint_Picc.png")
                plt.show()
                plt.close('all')

                pdb.set_trace()
                combined_df_new=pd.merge(site_forecast_data_percen,site_forecast_data_percen_2years,on='datetime')
                combined_df_new=pd.merge(combined_df_new,site_forecast_data_percen_6months,on='datetime')
                combined_df_new=pd.merge(combined_df_new,site_forecast_data_percen_cp,on='datetime')
                combined_df_new=pd.merge(combined_df_new,site_forecast_data_percen_pt,on='datetime')
                combined_df_new=pd.merge(combined_df_new,site_forecast_data_percen_pt_cp,on='datetime')

                fig = plt.figure(figsize=(20, 10))
                #plt.text(0.02, 0.9, textstr, fontsize=12, transform=plt.gcf().transFigure)
                plot = sns.jointplot(x="yhat", y="y", data=site_forecast_data, kind="hex").set_axis_labels("Predicted", "Measured")
                plot.ax_joint.plot(site_forecast_data.y,site_forecast_data.y, 'r-', linewidth = 1)
                plot.ax_joint.plot(site_forecast_data.y,site_forecast_data.y*1.25, 'b-.', linewidth = 0.8)
                plot.ax_joint.plot(site_forecast_data.y,site_forecast_data.y*0.75, 'b-.', linewidth = 0.8)
                min_x = min(site_forecast_data.y)
                max_x = max(site_forecast_data.y)
                plot.ax_joint.set_xlim(min_x,max_x)
                plot.ax_joint.set_ylim(min_x,max_x)
                #plot.fig.suptitle(textstr)
                plot.savefig(data_path+station_name+"_3year_month_forecast_Hexjoint_Picc.png")
                plt.close('all')

                fig = plt.figure(figsize=(20, 10))
                #plt.text(0.02, 0.9, textstr, fontsize=12, transform=plt.gcf().transFigure)
                plot = sns.jointplot(x="yhat", y="y", data=site_forecast_data_cp, kind="hex").set_axis_labels("Predicted", "Measured")
                plot.ax_joint.plot(site_forecast_data.y,site_forecast_data.y, 'r-', linewidth = 1)
                plot.ax_joint.plot(site_forecast_data.y,site_forecast_data.y*1.25, 'b-.', linewidth = 0.8)
                plot.ax_joint.plot(site_forecast_data.y,site_forecast_data.y*0.75, 'b-.', linewidth = 0.8)
                min_x = min(site_forecast_data.y)
                max_x = max(site_forecast_data.y)
                plot.ax_joint.set_xlim(min_x,max_x)
                plot.ax_joint.set_ylim(min_x,max_x)
                #plot.fig.suptitle(textstr)
                plot.savefig(data_path+station_name+"_3year_month_forecast_Hexjoint_Picc_cp.png")
                plt.close('all')

                # Now produce a boxplot
                #pdb.set_trace()
                #fig = plt.figure(figsize=(20, 10))
                my_pal = {"Prophet1": "b", "Prophet4": "orange", "Prophet5":"g", "Prophet6":"r"}
                ax = sns.boxplot(data=combined_df_new[['Prophet1','Prophet4','Prophet5','Prophet6']], orient="h",showmeans=True,meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black","markersize":"12"}, palette=my_pal)
                x_locator =ticker.FixedLocator([-100,-80,-60,-40,-20,0,-20,-40,60,80,100])
                ax.xaxis.set_minor_locator(x_locator)
                ax.grid(axis="x", color="black",alpha=.5, linewidth=1)
                labels = [item.get_text() for item in ax.get_yticklabels()]
                labels[0] = 'Vanilla'
                labels[1] = 'CP=10'
                labels[2] = 'PT'
                labels[3] = 'PT(CP=10)'
                ax.set_yticklabels(labels)

                #plt.title('Percentage Deviation for site types')
                plt.xlabel('% Deviation ')
                plt.xlim(-110, 130)
                plt.show()
                pdb.set_trace()
                plt.close('all')

                # Compare histograms of values
                first=sns.distplot(site_forecast_data['yhat'],  hist = False, kde=True, label='Vanilla',color='b')
                second=sns.distplot(site_forecast_data_cp['yhat'], hist = False, kde=True,label='CP=10',color='orange')
                third=sns.distplot(site_forecast_data_pt['yhat_pt'],  hist = False, kde=True, label='PT',color='g')
                fourth=sns.distplot(site_forecast_data_pt_cp['yhat_pt'],  hist = False, kde=True,label='PT(CP=10)',color='r')
                meas=sns.distplot(site_forecast_data['y'],  hist = False, kde=True,label='Measured',color='k', kde_kws={'linestyle':'--'})
                plt.legend(prop={'size': 12})
                plt.xlabel(r'NO2 $\mu g.m^{-3}$',labelsize=14)
                plt.ylabel('Density',labelsize=14)
                plt.show()


            except:
                print("Problem with ", station_name)
