import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning import LightningModule, Trainer
import pandas as pd
from pandas import Timestamp
import wandb
import os
import torch
import numpy as np
from tqdm import tqdm

import plotly.express as px
import plotly.graph_objects as go

from zipfile import ZipFile
import io
import functools

import re
import requests

import sys


class ISDValidationCallback(Callback):
    def __init__(self, isd_data_path: str = 'data/ISD/', start_date: str = '20200101', end_date: str = '20220101', area: str = None):
        """
        Initialize the ISD class.

        Parameters:
        - isd_data_path (str): The path to the ISD data directory. Default is 'data/ISD/'.
        - start_date (str): The start date in the format 'YYYYMMDD'. Default is '20200101'.
        - end_date (str): The end date in the format 'YYYYMMDD'. Default is '20220101'.
        """

        self.test_channels = ['si10', 'wdir10', 't2m', 'msl'] #'sp',


        self.start_time = pd.to_datetime(start_date)
        self.end_time: Timestamp = pd.to_datetime(end_date)

        self.data_root_path:  str = isd_data_path 
        self.isd_data_path: str = os.path.join(isd_data_path, 'preprocessed')

        self.area = area

        self.variable_dfs: dict = {}
        self.raw_index: dict = {}

        self.batch_size = None

        self.station_absolute_indices = {"Stations_id": [], "index": [], 'distance': []}
        self.station_absolute_indices_era5 = {"Stations_id": [], "index": [], 'distance': []}


    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:

        if stage == 'test':

            ### Prepare miscellaneous data
            self.years = trainer.datamodule.dataset_test.years
            if self.years[0] < self.start_time.year or self.years[-1] > self.end_time.year:
                raise ValueError(f"The years of the dataset {self.years} do not match the start and end dates {self.start_time.year} - {self.end_time.year}")
            
            if not trainer.datamodule.dataset_test.return_offset:
                raise ValueError("The dataset must return the offset for the ISD validation callback")
            

            self.returns_era = trainer.datamodule.dataset_test.return_era_original
            self.returns_cerra = trainer.datamodule.dataset_test.returns_cerra

            if self.area is None and self.returns_cerra:
                self.area = 'europe'
            elif self.area is None and self.returns_era:
                self.area = 'america'
            
            if self.area not in ['europe', 'america']:
                raise ValueError("The area must be either 'europe' or 'america'")

            if self.returns_era and self.returns_cerra:
                self.comparison_data = {'count': [], 'values': [], 'values_era': [], 'values_cerra': [], 'Stations_id': []}
            elif not self.returns_cerra:
                self.comparison_data = {'count': [], 'values': [], 'values_era': [], 'Stations_id': []}
            else:
                self.comparison_data = {'count': [], 'values': [], 'values_cerra': [], 'Stations_id': []}


            self.number_of_crops_per_sample = trainer.datamodule.dataset_test.number_of_crops_per_sample

            self.out_channels =  trainer.datamodule.dataset_test.out_channels
            self.in_channels = trainer.datamodule.dataset_test.in_channels
            self.channel_mapping = {self.out_channels[i]: i for i in range(len(self.out_channels))}
            self.channel_mapping_names = {pl_module.channel_names[self.out_channels[i]]: i for i in range(len(self.out_channels))}

            # Check if we have to convert the 10m wind to si10 and wdir10
            self.convert_10m_wind = ('si10' and 'wdir10' not in self.channel_mapping_names.keys()) and ('u10' and 'v10' in self.channel_mapping_names.keys())
            if self.convert_10m_wind:
                print("Converting 10m wind to si10 and wdir10")
                self.channel_mapping_names['si10'] = self.channel_mapping_names['u10']
                self.channel_mapping_names['wdir10'] = self.channel_mapping_names['v10']
                print("Self.channel_mapping_names", self.channel_mapping_names)
                
            self.is_sidechannel = trainer.datamodule.dataset_test.constant_channels

            
            self.variable_mean = trainer.datamodule.dataset_metrics['variable_mean'].reshape(1, -1, 1, 1)
            self.variable_std = trainer.datamodule.dataset_metrics['variable_std'].reshape(1, -1, 1, 1)


            self.complete_df = pd.DataFrame()
            for year in self.years:
                df_path = os.path.join(self.isd_data_path, str(year) + '.csv')
                if not os.path.exists(df_path):
                    raise ValueError(f"Missing data for year {year} at {df_path}")
                else:
                    self.complete_df = pd.concat([self.complete_df, pd.read_csv(df_path)], ignore_index=True)
            
            self.complete_df['DATE'] = pd.to_datetime(self.complete_df['DATE'])
            self.complete_df.drop(['WINDDIR_QUALITY', 'WINDSPEED_QUALITY', 'TEMPERATURE_QUALITY'], inplace=True, axis=1)
            self.complete_df.rename(columns={'STATION': 'Stations_id', 'TEMPERATURE': 't2m', 'MSLP': 'msl', 'WINDSPEED': 'si10', 'WINDDIR': 'wdir10', 'DATE': 'MESS_DATUM'}, inplace=True)

            ### Prepare the lat-lon maps:
            if self.area == 'america':
                print("Loading America Lat-Lon Map")
                self.lat_map, self.lon_map = torch.load(os.path.join(self.data_root_path, 'america_lat_lon.pt'), weights_only=False)
            else:
                print("Loading Europe Lat-Lon Map")
                self.lat_map, self.lon_map = torch.load(os.path.join(self.data_root_path, 'lat_lon_const.pt'), weights_only=False)


            ### Calculate the absolute indices for the stations
            self.index_df = self.complete_df.groupby('Stations_id').first().reset_index()[['LATITUDE', 'LONGITUDE', 'Stations_id']]
            stations = self.index_df.apply(lambda x: (x['LATITUDE'], x['LONGITUDE'], x['Stations_id']), axis=1)
            station_absolute_indices = stations.apply(lambda x: (self.find_nearest_lat_lon_index(self.lat_map, self.lon_map, x[0], x[1]), x[2])).to_list()
            self.station_absolute_indices['index'] = [x[0][0:2] for x in station_absolute_indices]
            self.station_absolute_indices['distance'] = [x[0][2] for x in station_absolute_indices]
            self.station_absolute_indices['Stations_id'] = [x[1] for x in station_absolute_indices]
            self.station_absolute_indices = pd.DataFrame(self.station_absolute_indices)

            len_tmp = len(self.station_absolute_indices)
            self.station_absolute_indices = self.station_absolute_indices[self.station_absolute_indices['distance'] <= 0.005]
            print(f"Removed {len_tmp - len(self.station_absolute_indices)} stations due to distance > 0.005. {len(self.station_absolute_indices)} stations remaining.")

                       
            


    def on_test_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        offsets = batch[2]
        size = outputs.shape[2]
        batch_size = outputs.shape[0]

        if self.batch_size is None:
            self.batch_size = batch_size

        cerra_data = batch[1][0] if self.is_sidechannel else batch[1]

        cerra_data = cerra_data.cpu() * self.variable_std[:, self.out_channels] + self.variable_mean[:, self.out_channels]
        output_data = outputs.cpu() 

        if self.returns_era and self.returns_cerra:
            era_data = batch[3].cpu() * self.variable_std[:, self.in_channels] + self.variable_mean[:, self.in_channels]
        

        for i, row in self.station_absolute_indices.iterrows():
            absolute_index = row['index']

            for j in range(batch_size):
                offset = (offsets[0][j], offsets[1][j])
                if (offset[0] <= absolute_index[0] and absolute_index[0] < offset[0] + size 
                    and offset[1] <= absolute_index[1] and absolute_index[1] < offset[1] + size):

                    self.comparison_data['values'].append(output_data[j, :, absolute_index[0] - offset[0], absolute_index[1] - offset[1]].clone())
                    if self.returns_cerra:
                        self.comparison_data['values_cerra'].append(cerra_data[j, :, absolute_index[0] - offset[0], absolute_index[1] - offset[1]].clone())
                    else:
                        self.comparison_data['values_era'].append(cerra_data[j, :, absolute_index[0] - offset[0], absolute_index[1] - offset[1]].clone())

                    self.comparison_data['count'].append(batch_idx * self.batch_size + j)
                    self.comparison_data['Stations_id'].append(row['Stations_id'])

                    if self.returns_era and self.returns_cerra:
                        self.comparison_data['values_era'].append(era_data[j, :, absolute_index[0] - offset[0], absolute_index[1] - offset[1]].clone())


    
    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        print("Calculating ISD Scores")
        complete_data = torch.stack(self.comparison_data['values']).cpu().numpy()

        if self.returns_cerra:
            complete_data_cerra = torch.stack(self.comparison_data['values_cerra']).cpu().numpy()

        if self.returns_era:
            complete_data_era = torch.stack(self.comparison_data['values_era']).cpu().numpy()


        if self.convert_10m_wind:
            si10 = np.sqrt(complete_data[:, self.channel_mapping_names['u10']]**2 + complete_data[:, self.channel_mapping_names['v10']]**2)
            wdir10 = np.arctan2(complete_data[:, self.channel_mapping_names['u10']],complete_data[:, self.channel_mapping_names['v10']])
            wdir10 = np.mod(180.0 + np.rad2deg(wdir10), 360.0)

            complete_data[:, self.channel_mapping_names['si10']] = si10
            complete_data[:, self.channel_mapping_names['wdir10']] = wdir10
            
            if self.returns_cerra:
                si10 = np.sqrt(complete_data_cerra[:, self.channel_mapping_names['u10']]**2 + complete_data_cerra[:, self.channel_mapping_names['v10']]**2)
                wdir10 = np.arctan2(complete_data_cerra[:, self.channel_mapping_names['u10']],complete_data_cerra[:, self.channel_mapping_names['v10']])
                wdir10 = np.mod(180.0 + np.rad2deg(wdir10), 360.0)

                complete_data_cerra[:, self.channel_mapping_names['si10']] = si10
                complete_data_cerra[:, self.channel_mapping_names['wdir10']] = wdir10

            if self.returns_era:
                si10 = np.sqrt(complete_data_era[:, self.channel_mapping_names['u10']]**2 + complete_data_era[:, self.channel_mapping_names['v10']]**2)
                wdir10 = np.arctan2(complete_data_era[:, self.channel_mapping_names['u10']],complete_data_era[:, self.channel_mapping_names['v10']])
                wdir10 = np.mod(180.0 + np.rad2deg(wdir10), 360.0)

                complete_data_era[:, self.channel_mapping_names['si10']] = si10
                complete_data_era[:, self.channel_mapping_names['wdir10']] = wdir10


        channels = [x for x in self.test_channels if x in self.channel_mapping_names.keys()]

        print(f"Channels: {channels}")

        for channel in channels:
            self.comparison_data[channel] = complete_data[:, self.channel_mapping_names[channel]]
            
            if self.returns_cerra:
                self.comparison_data[channel + '_cerra'] = complete_data_cerra[:, self.channel_mapping_names[channel]]

            if self.returns_era:
                self.comparison_data[channel + '_era'] = complete_data_era[:, self.channel_mapping_names[channel]]
                
        
        df = pd.DataFrame(self.comparison_data)


        
        if self.returns_era and self.returns_cerra:
            cerra_channels = [f'{x}_cerra' for x in channels]
            era_channels = [f'{x}_era' for x in channels]


            df1 = df.copy().drop(columns=[*cerra_channels, *era_channels])
            df1['source'] = 'ML'

            df2 = df.copy().drop(columns=[*channels, *era_channels])
            df2.rename(columns={cerra_channels[i]: channels[i] for i in range(len(channels))}, inplace=True)
            df2['source'] = 'CERRA'

            df3 = df.copy().drop(columns=[*channels, *cerra_channels])
            df3.rename(columns={era_channels[i]: channels[i] for i in range(len(channels))}, inplace=True)
            df3['source'] = 'ERA5 (bicubic)'


            df = pd.concat([df1, df2, df3], ignore_index=True)
        
        elif self.returns_cerra and not self.returns_era:
            cerra_channels = [f'{x}_cerra' for x in channels]
            df1 = df.copy().drop(columns=cerra_channels)
            df1['source'] = 'ML'

            df2 = df.copy().drop(columns=channels)
            df2.rename(columns={cerra_channels[i]: channels[i] for i in range(len(channels))}, inplace=True)
            df2['source'] = 'CERRA'

            df = pd.concat([df1, df2], ignore_index=True)

        else:
            era_channels = [f'{x}_era' for x in channels]
            df1 = df.copy().drop(columns=era_channels)
            df1['source'] = 'ML'

            df2 = df.copy().drop(columns=channels)
            df2.rename(columns={era_channels[i]: channels[i] for i in range(len(channels))}, inplace=True)
            df2['source'] = 'ERA5 (bicubic)'

            df = pd.concat([df1, df2], ignore_index=True)



        df['sample_index'] = df['count'] // self.number_of_crops_per_sample
        df['time_offset'] = df['sample_index'] * 3
        df['time'] = pd.to_datetime(df['time_offset'], unit='h', origin=pd.Timestamp(f"{self.years[0]}-01-01"))
        
        df.drop(columns=['values'], inplace=True)
        if self.returns_cerra:
            df.drop(columns=['values_cerra'], inplace=True)
        if self.returns_era:
            df.drop(columns=['values_era'], inplace=True)

        print(df.columns)
        print(self.complete_df.columns)
        

        ### Convert the temperature from collected data to Celsius
        df['t2m'] = df['t2m'] - 273.15
        df['msl'] = df['msl'] / 100
        
        df_test = df.merge(self.complete_df, left_on=['time', 'Stations_id'], right_on=['MESS_DATUM', 'Stations_id'], how='left', suffixes=('', '_isd'))

        df_test = df_test.drop(columns=['sample_index', 'time_offset', 'count', 'MESS_DATUM'])
       

        csv_path = os.path.join(self.isd_data_path, 'test_csv_isd_merged.csv')
        df_test.to_csv(csv_path, index=False)
        wandb.save(csv_path)

        self.log_scores_to_wandb(df_test)
        self.plot_station_scores(df_test)

    
    def log_scores_to_wandb(self, df):

        for source in df.source.unique():

            plot_df = df.copy()
            plot_df = plot_df[plot_df['source'] == source]
            log_results = {}

            prefix = '' if source == 'ML' else f'{source}_'

            for channel in self.test_channels:
                if channel in self.channel_mapping_names.keys():
                    
                    plot_df[channel + '_ae'] = abs(plot_df[channel] - plot_df[channel +'_isd'])
                    plot_df[channel + '_se'] = plot_df[channel + '_ae']**2

                    log_results[prefix + 'isd_' + channel + '_ae'] = plot_df[channel + '_ae'].mean(numeric_only=True)
                    log_results[prefix + 'isd_' + channel + '_se'] = plot_df[channel + '_se'].mean(numeric_only=True)


            plot_df['wdir10_ae'] = [min(abs((a % 360)-b), 360-abs((a % 360)-b)) for a, b in zip(plot_df['wdir10'], plot_df['wdir10_isd'])]
            plot_df['wdir10_se'] = plot_df['wdir10_ae']**2

            log_results[prefix + 'isd_' + 'wdir10' + '_ae'] = plot_df['wdir10' + '_ae'].mean(numeric_only=True)
            log_results[prefix + 'isd_' + 'wdir10' + '_se'] = plot_df['wdir10' + '_se'].mean(numeric_only=True)

            if source == 'ML':
                wandb.log(log_results)
        

        ### Plot the performance difference
        plot_df = df.copy()
        plot_results = {'source': [], 'metric': [], 'mae': [], 'mse': []}

        for channel in self.test_channels:
            if channel in self.channel_mapping_names.keys():
                
                plot_df[channel + '_ae'] = abs(plot_df[channel] - plot_df[channel +'_isd'])
                plot_df[channel + '_se'] = plot_df[channel + '_ae']**2

        
        plot_df = plot_df.groupby(['source']).mean(numeric_only=True).reset_index()

        for source in plot_df.source.unique():
            for channel in self.test_channels:
                if channel in self.channel_mapping_names.keys():
                    plot_results['source'].append(source)
                    plot_results['metric'].append(channel)
                    plot_results['mae'].append(plot_df[plot_df['source'] == source][channel + '_ae'].mean(numeric_only=True))
                    plot_results['mse'].append(plot_df[plot_df['source'] == source][channel + '_se'].mean(numeric_only=True))

        results_df = pd.DataFrame(plot_results)

        mae_perf_diff = []
        mse_perf_diff = []
        if self.returns_era:
            for source in plot_df.source.unique():

                comparison_mae = results_df[results_df['source'] == 'ERA5 (bicubic)']['mae'].values
                comparison_mse = results_df[results_df['source'] == 'ERA5 (bicubic)']['mse'].values

                mae = results_df[results_df['source'] == source]['mae'].values
                mse = results_df[results_df['source'] == source]['mse'].values

                mae_perf_diff.extend( - (mae - comparison_mae) / comparison_mae)
                mse_perf_diff.extend( - (mse - comparison_mse) / comparison_mse)

        results_df['mae_perf_diff'] = mae_perf_diff
        results_df['mse_perf_diff'] = mse_perf_diff

        fig = px.bar(results_df, x='source', y=['mae', 'mse'], facet_col='metric', title='Error against ISD-Dataset', barmode='group', width=1200, height=600)
        wandb.log({'isd_performance': fig})

        fig = px.bar(results_df, x='source', y=['mae_perf_diff', 'mse_perf_diff'], facet_col='metric', title='Relative Performance to ERA5 (Bicubic) for Different Datasets against ISD-Dataset', barmode='group', width=1200, height=600)
        wandb.log({'isd_performance_diff': fig})

        

    
    def plot_station_scores(self, df, plot_channel='t2m'):
        plot_df = df.copy()
        plot_df = plot_df[plot_df['source'] == 'ML']

        plot_channel_full = plot_channel + '_ae'
        
        for channel in self.test_channels:
            if channel in self.channel_mapping_names.keys():
                
                plot_df[channel + '_ae'] = abs(plot_df[channel] - plot_df[channel +'_isd'])
                plot_df[channel + '_se'] = plot_df[channel + '_ae']**2

        plot_df = plot_df.groupby('Stations_id').mean(numeric_only=True).reset_index()
        
        plot_df = plot_df.merge(self.index_df, on='Stations_id', how='left')

        fig = go.Figure()

        plot = px.scatter_geo(plot_df, lat='LATITUDE_x', lon='LONGITUDE_x', hover_data=['LATITUDE_x', 'LONGITUDE_x','ELEVATION', 'Stations_id'], color=plot_channel_full, title=f'Performance on ISD Station Data for {plot_channel}')
        fig = go.Figure(plot)
        fig.update_layout(width=1300, height=800)
        fig.update_geos(
            resolution=110,
            #center={'lat':51, 'lon': 10},
            showcoastlines=True, coastlinecolor="RebeccaPurple",
            showland=True, landcolor="LightGreen",
            showocean=True, oceancolor="LightBlue",
            showlakes=True, lakecolor="Blue",
            showrivers=True, rivercolor="Blue",
            lonaxis = dict(
                    showgrid = True,
                    gridwidth = 0.2,
                    gridcolor = '#737272',
                    range= [ -55.0, 68.0 ] if self.area == 'europe' else [-168.0, -35.0 ],
                    dtick = 1
                ),
                lataxis = dict (
                    showgrid = True,
                    gridwidth = 0.2,
                    gridcolor = '#737272',
                    range= [ 23.0, 75.0 ],
                    dtick = 1
                )
        )

        #fig.write_image('station_scores.png')
        wandb.log({'isd_station_scores': fig})
        
        
    def find_nearest_lat_lon_index(self, lat, lon, target_lat, target_lon):
        
        lat_diff = (lat - target_lat)
        lon_diff = (lon - target_lon)
        distance = lat_diff**2 + lon_diff**2
        index = distance.argmin(keepdims=True)

        index_lat = index // lat.shape[1]
        index_lon = index % lat.shape[1]

        return index_lat.item(), index_lon.item(), distance.min().item() #, lat[index_lat, index_lon].item(), lon[index_lat, index_lon].item(), target_lat, target_lon



if __name__ == '__main__':            
    validation_callback = ISDValidationCallback(isd_data_path='data/ISD/', start_date='20090101', end_date='20100101')
    validation_callback.setup(None, None, 'test')

    print(validation_callback.complete_df.head())