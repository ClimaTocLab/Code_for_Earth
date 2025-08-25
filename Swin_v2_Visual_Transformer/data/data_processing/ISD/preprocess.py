import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse

def preprocess_csv(file_path):
    selected_columns = ['STATION', 'DATE', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'WND', 'TMP', 'SLP']
    df = pd.read_csv(file_path, usecols=selected_columns)

    #Split and rescale the collumns with multiple values (e.g. Wind) into individual collumns (Wind speed, direction, quality)
    df['WINDDIR'] = df['WND'].str.split(',').str[0].astype(int)
    df['WINDDIR_QUALITY'] = df['WND'].str.split(',').str[1] #.astype(int)
    df['WIND_TYPE'] = df['WND'].str.split(',').str[2]
    df['WINDSPEED'] = df['WND'].str.split(',').str[3].astype(float) / 10
    df['WINDSPEED_QUALITY'] = df['WND'].str.split(',').str[4]

    df['TEMPERATURE'] = df['TMP'].str.split(',').str[0].astype(float) / 10
    df['TEMPERATURE_QUALITY'] = df['TMP'].str.split(',').str[1]

    df['MSLP'] = df['SLP'].str.split(',').str[0].astype(float) / 10
    df['MSLP_QUALITY'] = df['SLP'].str.split(',').str[1]


    #Rescale the remaining parameters to their correct range
    df['ELEVATION'] = df['ELEVATION'].astype(int)
    
    #Drop the duplicate collumns
    df = df.drop(columns=['WND', 'TMP', 'SLP'])

    #Replace missing data with NaN
    df['WINDDIR'] = df['WINDDIR'].replace(999, np.nan)
    df['WINDSPEED'] = df['WINDSPEED'].replace(999.9, np.nan)
    df['TEMPERATURE'] = df['TEMPERATURE'].replace(999.9, np.nan)
    df['MSLP'] = df['MSLP'].replace(9999.9, np.nan)
    df['ELEVATION'] = df['ELEVATION'].replace(9999, np.nan)
    df['LATITUDE'] = df['LATITUDE'].replace(99.999, np.nan)
    df['LONGITUDE'] = df['LONGITUDE'].replace(999.999, np.nan)

    #Change dtype of DATE to datetime
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%dT%H:%M:%S')

    #Drop rows wich don't correlate with a full hour
    df = df[df['DATE'].dt.minute == 0]

    return df


def filter_by_location(df, lat_min, lat_max, lon_min, lon_max):
    df = df[(df['LATITUDE'] >= lat_min) & (df['LATITUDE'] <= lat_max) & (df['LONGITUDE'] >= lon_min) & (df['LONGITUDE'] <= lon_max)]
    return df

def filter_hours(df, hours: list = [0, 3, 6, 9, 12, 15, 18, 21]):
    df = df[df['DATE'].dt.hour.isin(hours) & (df['DATE'].dt.minute == 0)]
    
    return df



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process ISD CSV files.')
    parser.add_argument('folder', type=str, help='Root path to the directory containing CSV files')
    args = parser.parse_args()
    root_path = args.folder

    files = os.listdir(root_path)

    file_path = root_path + '.csv'
    print(f"Processing files in {root_path}")
    print(f"Saving to {file_path}")

    full_df = pd.DataFrame()

    for file in tqdm(files):
        if file.endswith('.csv'):
            try:
                df = preprocess_csv(os.path.join(root_path, file))
            except Exception as e:
                print(f'Error reading file {file}: {e}')
                continue
            
            df = filter_hours(df, [0, 3, 6, 9, 12, 15, 18, 21 ])
            
            full_df = pd.concat([full_df, df])
    
    
    full_df.to_csv(file_path, index=False)
        