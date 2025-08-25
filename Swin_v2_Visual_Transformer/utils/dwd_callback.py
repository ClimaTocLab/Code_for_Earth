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


class DWDValidationCallback(Callback):
    def __init__(self, cdc_data_path: str = 'data/CDC/', start_date: str = '20200101', end_date: str = '20220101'):
        """
        Initialize the DWD class.

        Parameters:
        - cdc_data_path (str): The path to the CDC data directory. Default is 'data/CDC/'.
        - start_date (str): The start date in the format 'YYYYMMDD'. Default is '20200101'.
        - end_date (str): The end date in the format 'YYYYMMDD'. Default is '20220101'.
        """

        self.data_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"
        self.test_channels = ['si10', 'wdir10', 't2m', 'sp', 'msl']

        self.variable_paths = ['wind', 'air_temperature', 'pressure']

        self.start_time = pd.to_datetime(start_date)
        self.end_time: Timestamp = pd.to_datetime(end_date)

        self.cdc_data_path: str = cdc_data_path
        self.data_root_path: str = cdc_data_path[:-4]

        self.variable_dfs: dict = {}
        self.raw_index: dict = {}

        self.batch_size = None

        self.station_absolute_indices = {"Stations_id": [], "index": [], 'distance': []}
        self.station_absolute_indices_era5 = {"Stations_id": [], "index": [], 'distance': []}


    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:

        if stage == 'test':

            ### Prepare miscellaneous data
            self.years = trainer.datamodule.dataset_test.years
            self.returns_era = trainer.datamodule.dataset_test.return_era_original
            if self.years[0] < self.start_time.year or self.years[-1] > self.end_time.year:
                raise ValueError(f"The years of the dataset {self.years} do not match the start and end dates {self.start_time.year} - {self.end_time.year}")
            
            if not trainer.datamodule.dataset_test.return_offset:
                raise ValueError("The dataset must return the offset for the DWD validation callback")
            
            if self.returns_era:
                self.comparison_data = {'count': [], 'values': [], 'values_era': [], 'values_cerra': [], 'Stations_id': []}
            else:
                self.comparison_data = {'count': [], 'values': [], 'values_cerra': [], 'Stations_id': []}


            self.number_of_crops_per_sample = trainer.datamodule.dataset_test.number_of_crops_per_sample

            self.out_channels = trainer.datamodule.dataset_test.out_channels
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


            ### Prepare the data from the DWD
            index_csv_path = os.path.join(self.cdc_data_path, f'index_{self.start_time.strftime("%Y%m%d")}_{self.end_time.strftime("%Y%m%d")}.csv')
            if os.path.exists(index_csv_path):
                self.index_df = pd.read_csv(index_csv_path)
            else:
                self.index_df = self.download_index_df()
                self.index_df.to_csv(index_csv_path, index=False)
            
            complete_csv_path = os.path.join(self.cdc_data_path, f'complete_{self.start_time.strftime("%Y%m%d")}_{self.end_time.strftime("%Y%m%d")}.csv')
            
            if os.path.exists(complete_csv_path):
                self.complete_df = pd.read_csv(complete_csv_path)
                self.complete_df['MESS_DATUM'] = pd.to_datetime(self.complete_df['MESS_DATUM'])
            else:
                self.download_data(self.index_df)
                self.complete_df = self.open_zip_files(self.index_df)
                self.complete_df.to_csv(complete_csv_path, index=False)

            self.complete_df.rename(columns={'STATIONS_ID': 'Stations_id', 'TT_TU': 't2m', 'RF_TU': 'relative_humidity', 'P0': 'sp', 'P': 'msl', 'F': 'si10', 'D': 'wdir10'}, inplace=True)

            ### Prepare the lat-lon maps:
            self.lat_map, self.lon_map = torch.load(os.path.join(self.cdc_data_path, 'lat_lon_const.pt'), weights_only=False)
            self.era5_lat_map, self.era5_lon_map = torch.load(os.path.join(self.cdc_data_path, 'era5_lat_lon_const.pt'), weights_only=False)

            ### Calculate the absolute indices for the stations
            stations = self.index_df.apply(lambda x: (x['geoBreite'], x['geoLaenge'], x['Stations_id']), axis=1)
            station_absolute_indices = stations.apply(lambda x: (self.find_nearest_lat_lon_index(self.lat_map, self.lon_map, x[0], x[1]), x[2])).to_list()
            self.station_absolute_indices['index'] = [x[0][0:2] for x in station_absolute_indices]
            self.station_absolute_indices['distance'] = [x[0][2] for x in station_absolute_indices]
            self.station_absolute_indices['Stations_id'] = [x[1] for x in station_absolute_indices]
            self.station_absolute_indices = pd.DataFrame(self.station_absolute_indices)

            station_absolute_indices_era5 = stations.apply(lambda x: (self.find_nearest_lat_lon_index(self.era5_lat_map, self.era5_lon_map, x[0], x[1]), x[2])).to_list()
            self.station_absolute_indices_era5['index'] = [x[0][0:2] for x in station_absolute_indices_era5]
            self.station_absolute_indices_era5['distance'] = [x[0][2] for x in station_absolute_indices_era5]
            self.station_absolute_indices_era5['Stations_id'] = [x[1] for x in station_absolute_indices_era5]
            self.station_absolute_indices_era5 = pd.DataFrame(self.station_absolute_indices_era5)


    def on_test_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        offsets = batch[2]
        size = outputs.shape[2]
        batch_size = outputs.shape[0]

        if self.batch_size is None:
            self.batch_size = batch_size

        cerra_data = batch[1][0] if self.is_sidechannel else batch[1]

        cerra_data = cerra_data.cpu() * self.variable_std[:, self.out_channels] + self.variable_mean[:, self.out_channels]
        output_data = outputs.cpu() 

        if self.returns_era:
            era_data = batch[3].cpu() * self.variable_std[:, self.in_channels] + self.variable_mean[:, self.in_channels]
        

        for i, row in self.station_absolute_indices.iterrows():
            absolute_index = row['index']

            for j in range(batch_size):
                offset = (offsets[0][j], offsets[1][j])
                if (offset[0] <= absolute_index[0] and absolute_index[0] < offset[0] + size 
                    and offset[1] <= absolute_index[1] and absolute_index[1] < offset[1] + size):

                    self.comparison_data['values'].append(output_data[j, :, absolute_index[0] - offset[0], absolute_index[1] - offset[1]].clone())
                    self.comparison_data['values_cerra'].append(cerra_data[j, :, absolute_index[0] - offset[0], absolute_index[1] - offset[1]].clone())
                    self.comparison_data['count'].append(batch_idx * self.batch_size + j)
                    self.comparison_data['Stations_id'].append(row['Stations_id'])

                    if self.returns_era:
                        self.comparison_data['values_era'].append(era_data[j, :, absolute_index[0] - offset[0], absolute_index[1] - offset[1]].clone())


    
    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        print("Calculating DWD Scores")
        complete_data = torch.stack(self.comparison_data['values']).cpu().numpy()
        complete_data_cerra = torch.stack(self.comparison_data['values_cerra']).cpu().numpy()

        if self.returns_era:
            complete_data_era = torch.stack(self.comparison_data['values_era']).cpu().numpy()

        if self.convert_10m_wind:
            si10 = np.sqrt(complete_data[:, self.channel_mapping_names['u10']]**2 + complete_data[:, self.channel_mapping_names['v10']]**2)
            wdir10 = np.arctan2(complete_data[:, self.channel_mapping_names['u10']],complete_data[:, self.channel_mapping_names['v10']])
            wdir10 = np.mod(180.0 + np.rad2deg(wdir10), 360.0)

            complete_data[:, self.channel_mapping_names['si10']] = si10
            complete_data[:, self.channel_mapping_names['wdir10']] = wdir10
            
           
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
            self.comparison_data[channel + '_cerra'] = complete_data_cerra[:, self.channel_mapping_names[channel]]

            if self.returns_era:
                self.comparison_data[channel + '_era'] = complete_data_era[:, self.channel_mapping_names[channel]]
        

        df = pd.DataFrame(self.comparison_data)


        
        if self.returns_era:
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
        
        else:
            cerra_channels = [f'{x}_cerra' for x in channels]
            df1 = df.copy().drop(columns=cerra_channels)
            df1['source'] = 'ML'

            df2 = df.copy().drop(columns=channels)
            df2.rename(columns={cerra_channels[i]: channels[i] for i in range(len(channels))}, inplace=True)
            df2['source'] = 'CERRA'

            df = pd.concat([df1, df2], ignore_index=True)


        df['sample_index'] = df['count'] // self.number_of_crops_per_sample
        df['time_offset'] = df['sample_index'] * 3
        df['time'] = pd.to_datetime(df['time_offset'], unit='h', origin=pd.Timestamp(f"{self.years[0]}-01-01"))
        df.drop(columns=['values', 'values_cerra'], inplace=True)

        if self.returns_era:
            df.drop(columns=['values_era'], inplace=True)
        

        ### Convert the temperature from collected data to Celsius
        df['t2m'] = df['t2m'] - 273.15
        df['sp'] = df['sp'] / 100
        df['msl'] = df['msl'] / 100
        
  
        df_test = df.merge(self.complete_df, left_on=['time', 'Stations_id'], right_on=['MESS_DATUM', 'Stations_id'], how='left', suffixes=('', '_dwd'))

        df_test = df_test.drop(columns=['sample_index', 'time_offset', 'count', 'MESS_DATUM', 'relative_humidity'])
        df_test = df_test.dropna(subset=['si10_dwd', 'wdir10_dwd', 't2m_dwd', 'sp_dwd', 'msl_dwd'])

        csv_path = os.path.join(self.cdc_data_path, 'test_csv_dwd_merged.csv')
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
                    
                    plot_df[channel + '_ae'] = abs(plot_df[channel] - plot_df[channel +'_dwd'])
                    plot_df[channel + '_se'] = plot_df[channel + '_ae']**2

                    log_results[prefix + 'dwd_' + channel + '_ae'] = plot_df[channel + '_ae'].mean()
                    log_results[prefix + 'dwd_' + channel + '_se'] = plot_df[channel + '_se'].mean()


            plot_df['wdir10_ae_c'] = [min(abs((a % 360)-b), 360-abs((a % 360)-b)) for a, b in zip(plot_df['wdir10'], plot_df['wdir10_dwd'])]
            plot_df['wdir10_se_c'] = plot_df['wdir10_ae_c']**2

            log_results[prefix + 'dwd_' + 'wdir10' + '_ae' + '_c'] = plot_df['wdir10' + '_ae' + '_c'].mean()
            log_results[prefix + 'dwd_' + 'wdir10' + '_se' + '_c'] = plot_df['wdir10' + '_se' + '_c'].mean()

            if source == 'ML':
                wandb.log(log_results)
        

        ### Plot the performance difference
        plot_df = df.copy()
        plot_results = {'source': [], 'metric': [], 'mae': [], 'mse': []}

        for channel in self.test_channels:
            if channel in self.channel_mapping_names.keys():
                
                plot_df[channel + '_ae'] = abs(plot_df[channel] - plot_df[channel +'_dwd'])
                plot_df[channel + '_se'] = plot_df[channel + '_ae']**2

        
        plot_df = plot_df.groupby(['source']).mean().reset_index()

        for source in plot_df.source.unique():
            for channel in self.test_channels:
                if channel in self.channel_mapping_names.keys():
                    plot_results['source'].append(source)
                    plot_results['metric'].append(channel)
                    plot_results['mae'].append(plot_df[plot_df['source'] == source][channel + '_ae'].mean())
                    plot_results['mse'].append(plot_df[plot_df['source'] == source][channel + '_se'].mean())

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

        fig = px.bar(results_df, x='source', y=['mae', 'mse'], facet_col='metric', title='Error against DWD-Dataset', barmode='group', width=1200, height=600)
        wandb.log({'dwd_performance': fig})

        fig = px.bar(results_df, x='source', y=['mae_perf_diff', 'mse_perf_diff'], facet_col='metric', title='Relative Performance to ERA5 (Bicubic) for Different Datasets against DWD-Dataset', barmode='group', width=1200, height=600)
        wandb.log({'dwd_performance_diff': fig})

        

    
    def plot_station_scores(self, df, plot_channel='t2m'):
        plot_df = df.copy()
        plot_df = plot_df[plot_df['source'] == 'ML']

        plot_channel_full = plot_channel + '_ae'
        
        for channel in self.test_channels:
            if channel in self.channel_mapping_names.keys():
                
                plot_df[channel + '_ae'] = abs(plot_df[channel] - plot_df[channel +'_dwd'])
                plot_df[channel + '_se'] = plot_df[channel + '_ae']**2

        plot_df = plot_df.groupby('Stations_id').mean(numeric_only=True).reset_index()
        
        plot_df = plot_df.merge(self.index_df, on='Stations_id', how='left')

        fig = go.Figure()

        plot = px.scatter_geo(plot_df, lat='geoBreite', lon='geoLaenge', hover_data=['Stationsname', 'Bundesland', 'Stationshoehe', 'Stations_id'], color=plot_channel_full, title=f'Performance on DWD Station Data for {plot_channel}')
        fig = go.Figure(plot)
        fig.update_layout(width=1000, height=1000)
        fig.update_geos(
            resolution=110,
            center={'lat':51, 'lon': 10},
            showcoastlines=True, coastlinecolor="RebeccaPurple",
            showland=True, landcolor="LightGreen",
            showocean=True, oceancolor="LightBlue",
            showlakes=True, lakecolor="Blue",
            showrivers=True, rivercolor="Blue",
            scope='europe',
            lonaxis = dict(
                    showgrid = True,
                    gridwidth = 0.2,
                    gridcolor = '#737272',
                    range= [ -7.0, 7.0 ],
                    dtick = 0.25
                ),
                lataxis = dict (
                    showgrid = True,
                    gridwidth = 0.2,
                    gridcolor = '#737272',
                    range= [ 45.0, 55.0 ],
                    dtick = 0.25
                )
        )

        #fig.write_image('station_scores.png')
        wandb.log({'dwd_station_scores': fig})
        
        

        
    def download_index_df(self):

        if not os.path.exists(self.cdc_data_path):
            os.makedirs(self.cdc_data_path)

            os.makedirs(os.path.join(self.cdc_data_path, 'raw'))
        
        self.variable_dfs = {}

        for variable in self.variable_paths:
            raw_data_path = os.path.join(self.cdc_data_path, 'raw', variable)

            if not os.path.exists(raw_data_path):
                os.makedirs(raw_data_path)

            url = f"{self.data_url}/{variable}/historical/"

            ### grab the Index file to get the list of files to download
            response = requests.get(url)
            
            if response.status_code != 200:
                raise Exception(f"Failed to download data for {variable}")
            
            self.raw_index[variable] = response.text
            index_file_name = re.findall(r'<a href="([^"]+\.txt)">', response.text)
            data_file_pattern = re.findall(r'<a href="([^"]+\.zip)">', response.text)[0]
            data_file_prefix = '_'.join(data_file_pattern.split('_')[:2])
            data_file_suffix = f"_{data_file_pattern.split('_')[-1]}"
            

            response = requests.get(url + index_file_name[0])
            
            index_path = os.path.join(raw_data_path, index_file_name[0])
            with open(index_path, 'w') as f:
                
                modified_text = re.sub(r' {2,}', ';', response.text)
                modified_text = re.sub(r'(?<=\d) (?=\d)|(?<=\d) (?=[A-Za-zäöüÄÖÜ])', ';', modified_text)

                modified_text_lines = modified_text.split('\n')
                modified_text = '\n'.join(modified_text_lines[2:])

                column_names = modified_text_lines[0].split(' ')

                f.write(';'.join(column_names))
                f.write(modified_text)

            df = pd.read_csv(index_path, sep=';', header=0, index_col=False)
            df['von_datum'] = pd.to_datetime(df['von_datum'], format='%Y%m%d')
            df['bis_datum'] = pd.to_datetime(df['bis_datum'], format='%Y%m%d')

            df = df[(df['von_datum'] <= self.start_time) & (df['bis_datum'] >= self.end_time)]

            #Only assign the first part of the filename, as the end_date may be different in the actual filename
            df[f'filename_{variable}'] = df.apply(lambda x: f"{data_file_prefix}_{x['Stations_id']:05d}_{x['von_datum'].strftime('%Y%m%d')}_", axis=1)

            #As the end_date in the index is different to that in the filename, we need to get the filename from the index
            df[f'filename_{variable}'] = df.apply(lambda x: self.get_filename_from_index(x, variable, self.raw_index), axis=1)

            df[variable] = True
            
            self.variable_dfs[variable] = df.drop(columns=['von_datum', 'bis_datum'])
            
        
        merged_df = pd.merge(self.variable_dfs['wind'] , self.variable_dfs['air_temperature'], on=['Stations_id', 'Stationshoehe', 'geoBreite', 'geoLaenge', 'Stationsname', 'Bundesland'], how='outer')
        merged_df = pd.merge(merged_df, self.variable_dfs['pressure'], on=['Stations_id', 'Stationshoehe', 'geoBreite', 'geoLaenge', 'Stationsname', 'Bundesland'], how='outer')
        merged_df = merged_df.dropna(axis='rows', subset=[*self.variable_paths, *['filename_' + variable for variable in self.variable_paths]])
        merged_df = merged_df.drop(columns=['wind', 'air_temperature', 'pressure'])
        merged_df.reset_index(drop=True, inplace=True)
        merged_df = merged_df.astype({'Stations_id': 'int64', 'Stationshoehe': 'int64', 'geoBreite': 'float64', 'geoLaenge': 'float64', 'Stationsname': 'str', 'Bundesland': 'str', 'filename_wind': 'str', 'filename_air_temperature': 'str', 'filename_pressure': 'str'})
        return merged_df
    
    def get_filename_from_index(self, row, variable, raw_index):
                matches = re.findall(r'<a href="(' + row[f'filename_{variable}'] + '[^"]+\.zip)">', raw_index[variable])
                return matches[0] if matches else None
    

    def download_data(self, df: pd.DataFrame):
        for index, row in tqdm(df.iterrows(), total=len(df), desc='Downloading DWD Data'):
            for variable in self.variable_paths:

                filename = row[f'filename_{variable}']
                file_path = os.path.join(self.cdc_data_path, 'raw', variable, filename)
                if not os.path.exists(file_path):
                    url = f"{self.data_url}/{variable}/historical/{filename}"
                    response = requests.get(url)

                    if response.status_code != 200:
                        raise Exception(f"Failed to download data for {url}")
                    
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
            

    def open_zip_files(self, df: pd.DataFrame):
        complete_df = pd.DataFrame()

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Reading DWD Data from ZIP Files"):
            dfs = []

            for variable in self.variable_paths:
                filename = row[f'filename_{variable}']
                file_path = os.path.join(self.cdc_data_path, 'raw', variable, filename)
                dfs.append(self.read_zip_file(file_path))
            
            merged_df = functools.reduce(lambda left, right: pd.merge(left,right,on=['STATIONS_ID', 'MESS_DATUM']), dfs)
            merged_df['MESS_DATUM'] = pd.to_datetime(merged_df['MESS_DATUM'], format='%Y%m%d%H')
            merged_df = merged_df.replace(-999.0, None)
            merged_df = merged_df.drop(columns=['QN_3', 'QN_9', 'QN_8'])
            merged_df = merged_df.astype({'F': 'float64','D': 'float64', 'TT_TU': 'float64', 'RF_TU': 'float64', 'P': 'float64', 'P0': 'float64'})
            

            #Filter the data
            merged_df = merged_df[merged_df['MESS_DATUM'].dt.hour.isin([0, 3, 6, 9, 12, 15, 18, 21])]
            merged_df = merged_df[(merged_df['MESS_DATUM'] >= self.start_time) & (merged_df['MESS_DATUM'] <= self.end_time)]


            complete_df = pd.concat([complete_df, merged_df], ignore_index=True)

        return complete_df
            
            


    def read_zip_file(self, file_path: str):
        """
        Reads a zip file and returns the relevant timeseries data as a pandas DataFrame.

        Parameters:
        file_path (str): The path to the zip file.

        Returns:
        pandas.DataFrame: The contents as a DataFrame.
        """
        
        with ZipFile(file_path, 'r') as z:
            files = z.namelist()
            files = list(filter(lambda x: 'produkt_' in x, files))

            assert len(files)==1, f"File {file_path} contains {len(files)} files, expected 1"

            with z.open(files[0]) as f:
                #Read the file, convert it to a valid CSV and then read it with pandas
                data_str = f.read().decode('utf-8')
                valid_csv = re.sub(r'\s*;\s*eor', '\n', re.sub(r'\s*;\s*', ';', re.sub(r'\s+', ' ', data_str.strip()))) 
                df = pd.read_csv(io.StringIO(valid_csv), sep=';', header=0, index_col=False)
                return df
    
    def find_nearest_lat_lon_index(self, lat, lon, target_lat, target_lon):
        
        lat_diff = (lat - target_lat)
        lon_diff = (lon - target_lon)
        distance = lat_diff**2 + lon_diff**2
        index = distance.argmin(keepdims=True)

        index_lat = index // lat.shape[1]
        index_lon = index % lat.shape[1]

        return index_lat.item(), index_lon.item(), distance.min().item() 

if __name__ == '__main__':            
    validation_callback = DWDValidationCallback()
    validation_callback.setup(None, None, 'test')

    print(validation_callback.complete_df.head())