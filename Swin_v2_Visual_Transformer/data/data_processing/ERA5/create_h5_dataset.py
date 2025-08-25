import os
import sys
import argparse
import subprocess


import h5py
from netCDF4 import Dataset as DS
import numpy as np
import torch

channel_names = [ 'si10', 'wdir10', 't2m', 'sp', 'msl', 't850', 'u1000', 'v1000', 'z1000', 'u850', 'v850', 'z850', 'u500', 'v500', 'z500', 't500', 'z50', 'r500', 'r850', 'tcwv']

parameter_ids= {
                'sp': 'var134', 'msl': 'var151', 't2m': 'var167', 'u10': 'var165', 'v10': 'var166', 'tcwv': 'var137',
                'z1000': 'var129', 'u1000': 'var131', 'v1000': 'var132',
                'z50': 'var129',
                'z500': 'var129', 'u500': 'var131', 'v500': 'var132', 't500': 'var130', 'r500': 'var157', 'z850': 'var129', 'u850': 'var131', 'v850': 'var132', 't850': 'var130', 'r850': 'var157'
            }

parameter_ids_backup = {
                'sp': 'sp', 'msl': 'msl', 't2m': '2t', 'u10': '10u', 'v10': '10v', 'tcwv': 'tcwv',
                'z1000': 'z', 'u1000': 'u', 'v1000': 'v',
                'z50': 'z',
                'z500': 'z', 'u500': 'u', 'v500': 'v', 't500': 't', 'r500': 'r', 'z850': 'z', 'u850': 'u', 'v850': 'v', 't850': 't', 'r850': 'r'
            }


def create_all_datasets(input_dir, gridfile_path, update=False, no_regrid: bool=False, mode: str='bicubic', windmode: str='combined', selected_year: int=None):
    """
    Create datasets for all available years and months in the given input directory.

    Args:
        input_dir (str): The path to the input directory containing the data.
        update (bool, optional): If True, update the dataset even if it already exists. 
            Defaults to False.

    Returns:
        None
    """
    years = os.listdir(input_dir)
    for year in years:

        if selected_year is None or int(year) == selected_year:
            print(f"Checking {year}")

            months = sorted(os.listdir(os.path.join(input_dir, year)))
            for month in months:
                files = os.listdir(os.path.join(input_dir, year, month))
                if len(files) < 4:
                    print(f"Skipping {year}-{month} due to missing files")
                    continue

                if int(year) == 2021 and int(month) > 6:
                    print(f"Skipping {year}-{month} due to no available CERRA data")
                    continue

                print(" ################################## ")
                print(f"Creating dataset for {year}-{month} \n")
                create_h5_dataset_month(os.path.join(input_dir, year, month), gridfile_path, update, not no_regrid, mode, windmode)
                print(" ################################## \n")

        else:
            print(f"Skipping {year} due to year selection")

        



def create_h5_dataset_month(input_dir, gridfile_path, update=False, regrid: bool=True, mode: str='bicubic', windmode: str='combined'):
    """
    Create an HDF5 dataset for a specific month.

    Args:
        input_dir (str): The directory containing the input files.
        update (bool, optional): If True, update the dataset even if it already exists. 
            Defaults to False.
        regrid (bool, optional): If True, regrid the data to the same grid as CERRA.
            Defaults to True.
        mode (str, optional): The interpolation mode for regridding. Options: bilinear, bicubic
            Defaults to 'bicubic'.
        windmode (str, optional): The wind mode to use. Options: combined, separate
            Defaults to 'combined'.

    Returns:
        None
    """

    n_variables = 20
    n_timesteps = None

    if regrid:
        y_dim = 1069
        x_dim = 1069
    else:
        y_dim = 261
        x_dim = 561
    
    year = input_dir.split('/')[-2]
    month = input_dir.split('/')[-1]
    
    if regrid:
        if mode == 'bilinear':
            output_file = f"{input_dir}/{year}{month}_bilinear.h5"
        else:
            output_file = f"{input_dir}/{year}{month}.h5"
    else:
        output_file = f"{input_dir}/{year}{month}_original.h5"

    if windmode == 'combined':
        pass
    elif windmode == 'separate':
        output_file = output_file.replace('.h5', '_separate.h5')
    else:
        raise ValueError("Invalid windmode. Options: combined, separate")

    files = ['upper.nc4', 'middle.nc4', 'lower.nc4', 'surface.nc4']


    if regrid:
        #Regrid the era5 data to the same grid as the CERRA data
        for file in files:
            file = file[:-4]
            if mode == 'bilinear':
                subprocess.run(['cdo', f"remapbil,{gridfile_path}", f"{input_dir}/{file}.grib", f"{input_dir}/{file}_regrid.grib"], check=True)
            else:
                subprocess.run(['cdo', f"remapbic,{gridfile_path}", f"{input_dir}/{file}.grib", f"{input_dir}/{file}_regrid.grib"], check=True)
            subprocess.run(['cdo', '-f', 'nc4', '-z', 'zip=9', '-copy', '-setgrid,' + gridfile_path,  f"{input_dir}/{file}_regrid.grib", f"{input_dir}/{file}_regrid.nc4"], check=True)
            os.remove(f"{input_dir}/{file}_regrid.grib")
            print(f"Converted {file}.grib to {file}_regrid.nc4")

    else:
        for file in files:
            file = file[:-4]
            subprocess.run(['cdo', '-f', 'nc4', '-z', 'zip=9', '-copy',  f"{input_dir}/{file}.grib", f"{input_dir}/{file}.nc4"], check=True)
            print(f"Converted {file}.grib to {file}.nc4")


    if regrid:
        lower_data = DS(f"{input_dir}/lower_regrid.nc4")
    else:
        lower_data = DS(f"{input_dir}/lower.nc4")

    n_timesteps = lower_data.dimensions['time'].size
    ids =  parameter_ids if 'var129' in lower_data.variables.keys() else parameter_ids_backup


    if not update and os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping.")
        return
    
    with h5py.File(output_file, 'w') as f:
        h5_dataset = f.create_dataset('data', (n_timesteps, n_variables, y_dim, x_dim), dtype='float32')
        print(f"Creating dataset with shape: {h5_dataset.shape}")
        

        print(f"Adding data from lower.nc4")
        try:
            h5_dataset[:, 6, :, :] = np.flip(lower_data[ids['u1000']][:, 0], [1])
        except IndexError as e:
            print(f"u1000 not found in {input_dir}/lower.nc4. Using backup_ids for file instead.")
            ids = parameter_ids_backup
            h5_dataset[:, 6, :, :] = np.flip(lower_data[ids['u1000']][:, 0], [1])

        h5_dataset[:, 7, :, :] = np.flip(lower_data[ids['v1000']][:, 0], [1])
        h5_dataset[:, 8, :, :] = np.flip(lower_data[ids['z1000']][:, 0], [1])
        
        del lower_data

        print(f"Adding data from surface_regrid.nc4")

        if regrid:
            surface_data = DS(f"{input_dir}/surface_regrid.nc4")
        else:
            surface_data = DS(f"{input_dir}/surface.nc4")
        
        #Convert u10 and v10 to si10 and wdir10 (wind speed and direction at 10m above ground level, as used by CERRA)
        if windmode == 'combined':
            si10 = np.sqrt(surface_data[ids['u10']][:] ** 2 + surface_data[ids['v10']][:] ** 2)
            wdir10 = np.arctan2(surface_data[ids['u10']][:], surface_data[ids['v10']][:])
            wdir10 = np.mod(180.0 + np.rad2deg(wdir10), 360.0)

            h5_dataset[:, 0, :, :] = np.flip(si10, [1])
            h5_dataset[:, 1, :, :] = np.flip(wdir10, [1])

        elif windmode == 'separate':
            h5_dataset[:, 0, :, :] = np.flip(surface_data[ids['u10']][:], [1])
            h5_dataset[:, 1, :, :] = np.flip(surface_data[ids['v10']][:], [1])

        else:
            raise ValueError("Invalid windmode. Options: combined, separate")

        h5_dataset[:, 3, :, :] = np.flip(surface_data[ids['sp']][:], [1])
        h5_dataset[:, 4, :, :] = np.flip(surface_data[ids['msl']][:], [1])
        h5_dataset[:, 19, :, :] = np.flip(surface_data[ids['tcwv']][:], [1])
        h5_dataset[:, 2, :, :] = np.flip(surface_data[ids['t2m']][:], [1])
        
        del surface_data

        print(f"Adding data from middle_regrid.nc4")
        if regrid:
            middle_data = DS(f"{input_dir}/middle_regrid.nc4")
        else:
            middle_data = DS(f"{input_dir}/middle.nc4")

        h5_dataset[:, 5, :, :] = np.flip(middle_data[ids['t850']][:, 1], [1])
        h5_dataset[:, 9, :, :] = np.flip(middle_data[ids['u850']][:, 1], [1])
        h5_dataset[:, 10, :, :] = np.flip(middle_data[ids['v850']][:, 1], [1])
        h5_dataset[:, 11, :, :] = np.flip(middle_data[ids['z850']][:, 1], [1])
        h5_dataset[:, 12, :, :] = np.flip(middle_data[ids['u500']][:, 0], [1])
        h5_dataset[:, 13, :, :] = np.flip(middle_data[ids['v500']][:, 0], [1])
        h5_dataset[:, 14, :, :] = np.flip(middle_data[ids['z500']][:, 0], [1])
        h5_dataset[:, 15, :, :] = np.flip(middle_data[ids['t500']][:, 0], [1])
        h5_dataset[:, 17, :, :] = np.flip(middle_data[ids['r500']][:, 0], [1])
        h5_dataset[:, 18, :, :] = np.flip(middle_data[ids['r850']][:, 1], [1])

        del middle_data

        print(f"Adding data from upper_regrid.nc4")
        if regrid:
            upper_data = DS(f"{input_dir}/upper_regrid.nc4")
        else:
            upper_data = DS(f"{input_dir}/upper.nc4")

        h5_dataset[:, 16, :, :] = np.flip(upper_data[ids['z50']][:, 0], [1])

        del upper_data


        for file in files:
            if regrid:
                os.remove(f"{input_dir}/{file[:-4]}_regrid.nc4")
            else:
                os.remove(f"{input_dir}/{file[:-4]}.nc4")
            print(f"Removed {file}_regrid.nc4")

def convert_const_to_pt(input_file):
    """
    Convert land-sea-mask and orography to pt format.'var

 '   Args:
        input_file (str): The path to the input file.

    Returns:
        None (Saves the output file to disk)
    """
    output_file = input_file.replace('.nc4', '.pt')
    print(f"Converting {input_file} to {output_file}")

    data = DS(input_file)
    n_timesteps = 1
    y_dim = 1069
    x_dim = 1069

    print("Creating dataset with shape: (2, 1069, 1069)")

    arr = np.empty((2, y_dim, x_dim), dtype=np.float32)
    arr[0] = np.flip(data['lsm'][:], [0])
    arr[1] = np.flip(data['orog'][:], [0])

    t = torch.tensor(arr, dtype=torch.float32)
    torch.save(t, output_file)
    

def main():
    parser = argparse.ArgumentParser(description='Create h5 dataset from CERRA data')
    parser.add_argument('input_dir', type=str, help='Directory with CERRA data')
    parser.add_argument('gridfile_path', type=str, default='/data/data_processing/ERA5/cerra_grid.txt', help='Path to the gridfile')
    parser.add_argument('--mode', type=str, default='bicubic', help='Interpolation mode for regridding. Options: bilinear, bicubic')
    parser.add_argument('--year', type=int, help='Select year for processing, e.g. 2020. If not specified, all years will be processed')
    parser.add_argument('--update', action='store_true', help='Update existing datasets')
    parser.add_argument('--no_regrid', action='store_true', help='Do not regrid the data to the same grid as CERRA')
    parser.add_argument('--wind', type=str, default='combined', help='Determines wether to use si10,wdir10 or u10,v10 for the 10m wind. Options: combined, separate')
    
    args = parser.parse_args()
    
    create_all_datasets(args.input_dir, args.gridfile_path, args.update, args.no_regrid, args.mode, args.wind, args.year)

if __name__ == "__main__":
    main()