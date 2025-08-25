import os
import argparse

import h5py
from netCDF4 import Dataset as DS
import numpy as np
import torch

channel_names = [ 'si10', 'wdir10', 't2m', 'sp', 'msl', 't850', 'u1000', 'v1000', 'z1000', 'u850', 'v850', 'z850', 'u500', 'v500', 'z500', 't500', 'z50', 'r500', 'r850', 'tcwv']

def create_all_datasets(input_dir, update=False, windmode: str='combined', selected_year:int =None):
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

        print(f"Checking {year}")
        if (selected_year is None and int(year) in range(2009, 2022)) or int(year) == selected_year:
            
            months = os.listdir(os.path.join(input_dir, year))
            for month in months:
                files = os.listdir(os.path.join(input_dir, year, month))
                if len(files) < 4:
                    print(f"Skipping {year}-{month} due to missing files")
                    continue

                print(" ################################## ")
                print(f"Creating dataset for {year}-{month} \n")
                create_h5_dataset_month(os.path.join(input_dir, year, month), update, windmode)
                print(" ################################## \n")

        else:
            print(f"Skipping {year} due to year selection")



def create_h5_dataset_month(input_dir, update=False, windmode: str='combined'):
    """
    Create an HDF5 dataset for a specific month.

    Args:
        input_dir (str): The directory containing the input files.
        update (bool, optional): If True, update the dataset even if it already exists. 
            Defaults to False.
        windmode (str, optional): The wind mode to use. Options: combined, separate
            Defaults to 'combined'.

    Returns:
        None
    """

    n_variables = 20
    n_timesteps = None

    y_dim = 1069
    x_dim = 1069
    
    year = input_dir.split('/')[-2]
    month = input_dir.split('/')[-1]
    
    if windmode == 'combined':
        output_file = f"{input_dir}/{year}{month}.h5"
    elif windmode == 'separate':
        output_file = f"{input_dir}/{year}{month}_separate.h5"
    else:
        raise ValueError("Invalid windmode. Options: combined, separate")

    files = ['upper.nc4', 'middle.nc4', 'lower.nc4', 'surface.nc4']

    lower_data = DS(f"{input_dir}/lower.nc4")
    n_timesteps = lower_data.dimensions['time'].size


    if not update and os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping.")
        return
    
    with h5py.File(output_file, 'w') as f:
        h5_dataset = f.create_dataset('data', (n_timesteps, n_variables, y_dim, x_dim), dtype='float32')
        print(f"Creating dataset with shape: {h5_dataset.shape}")

        print(f"Adding data from lower.nc4")
        h5_dataset[:, 6, :, :] = np.flip(lower_data['u'][:], [1])
        h5_dataset[:, 7, :, :] = np.flip(lower_data['v'][:], [1])
        h5_dataset[:, 8, :, :] = np.flip(lower_data['z'][:], [1])

        del lower_data

        print(f"Adding data from surface.nc4")
        surface_data = DS(f"{input_dir}/surface.nc4")

        if windmode == 'combined':
            h5_dataset[:, 0, :, :] = np.flip(surface_data['si10'][:], [1])
            h5_dataset[:, 1, :, :] = np.flip(surface_data['wdir10'][:], [1])

        elif windmode == 'separate':
            si10 = np.flip(surface_data['si10'][:], [1])
            wdir10 = np.flip(surface_data['wdir10'][:], [1])

            wdir10_rad = np.deg2rad(wdir10)
            u10 = -si10 * np.sin(wdir10_rad)
            v10 = -si10 * np.cos(wdir10_rad)


            h5_dataset[:, 0, :, :] = u10
            h5_dataset[:, 1, :, :] = v10

        else:
            raise ValueError("Invalid windmode. Options: combined, separate")
        
        h5_dataset[:, 3, :, :] = np.flip(surface_data['sp'][:], [1])
        h5_dataset[:, 4, :, :] = np.flip(surface_data['msl'][:], [1])
        h5_dataset[:, 19, :, :] = np.flip(surface_data['tciwv'][:], [1])
        
        try:
            h5_dataset[:, 2, :, :] = surface_data['t2m'][:]
        except IndexError as e:
            print(f"t2m not found in {input_dir}/surface.nc4. Using t2m from t2m.nc4 instead.")
            t2m_data = DS(f"{input_dir}/t2m.nc4")
            h5_dataset[:, 2, :, :] = np.flip(t2m_data['t2m'][:], [1])
            del t2m_data

        del surface_data

        print(f"Adding data from middle.nc4")
        middle_data = DS(f"{input_dir}/middle.nc4")
        h5_dataset[:, 5, :, :] = np.flip(middle_data['t'][:, 0], [1])
        h5_dataset[:, 9, :, :] = np.flip(middle_data['u'][:, 0], [1])
        h5_dataset[:, 10, :, :] = np.flip(middle_data['v'][:, 0], [1])
        h5_dataset[:, 11, :, :] = np.flip(middle_data['z'][:, 0], [1])
        h5_dataset[:, 12, :, :] = np.flip(middle_data['u'][:, 1], [1])
        h5_dataset[:, 13, :, :] = np.flip(middle_data['v'][:, 1], [1])
        h5_dataset[:, 14, :, :] = np.flip(middle_data['z'][:, 1], [1])
        h5_dataset[:, 15, :, :] = np.flip(middle_data['t'][:, 1], [1])
        h5_dataset[:, 17, :, :] = np.flip(middle_data['r'][:, 1], [1])
        h5_dataset[:, 18, :, :] = np.flip(middle_data['r'][:, 0], [1])

        del middle_data

        upper_data = DS(f"{input_dir}/upper.nc4")
        try:
            h5_dataset[:, 16, :, :] = np.flip(upper_data['z'][:], [1])
        except TypeError as e:
            h5_dataset[:, 16, :, :] = np.flip(upper_data['z'][:, 0], [1])


        del upper_data

def convert_const_to_pt(input_file):
    """
    Convert land-sea-mask and orography to pt format.

    Args:
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
    
def test_h5_file(file):
    with h5py.File(file, 'r') as f:
        data = f['data'][:]
        print(data.shape)

        print(data[0, 0, 0, 0])

        print(data[0, 7, 0, 0])

        print(data[0, 17, 0, 0])
        print(data[0, 18, 0, 0])

def main():
    parser = argparse.ArgumentParser(description='Create h5 dataset from CERRA data')
    parser.add_argument('input_dir', type=str, help='Directory with CERRA data')
    parser.add_argument('--update', action='store_true', help='Update existing datasets')
    parser.add_argument('--wind', type=str, default='combined', help='Determines wether to use si10,wdir10 or u10,v10 for the 10m wind. Options: combined, separate')
    parser.add_argument('--year', type=int, help='Select year for processing, e.g. 2020. If not specified, all years will be processed')
    # parser.add_argument("--start_year", type=int, default=2016)
    # parser.add_argument("--end_year", type=int, default=2017)
    args = parser.parse_args()

    create_all_datasets(args.input_dir, args.update, args.wind, args.year)

if __name__ == "__main__":
    main()