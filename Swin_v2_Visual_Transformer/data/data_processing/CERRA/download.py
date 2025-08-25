import cdsapi
import os
import argparse


def main(output_dir, start_year: int = 2016, end_year: int = 2019):

    client = cdsapi.Client()

    for year in range(start_year,end_year+1):
        for month in range(7,13):


            month_str = f"{month:02d}"
            year_str = str(year)

            dir = os.path.join(output_dir, year_str, month_str)
            if not os.path.exists(dir):
                os.makedirs(dir)

            retrieve_upper_level(client, year_str, month_str, dir)
            retrieve_middle_levels(client, year_str, month_str, dir)
            retrieve_lower_level(client, year_str, month_str, dir)
            retrieve_surface(client, year_str, month_str, dir)
            retrieve_t2m(client, year_str, month_str, dir)




def retrieve_middle_levels(c, year: str, month: str, dir: str):
    print(f"Retreiving middle levels for {year}-{month}")

    file = os.path.join(dir, "middle.nc4")

    if os.path.exists(file):
        print(f"File {file} already exists, skipping")
        return

    c.retrieve(
    'reanalysis-cerra-pressure-levels',
    {
        'format': 'netcdf',
        'variable': [
            'geopotential', 'relative_humidity', 'temperature',
            'u_component_of_wind', 'v_component_of_wind',
        ],
        'pressure_level': [
            '500', '850',
        ],
        'data_type': 'reanalysis',
        'product_type': 'analysis',
        'year': year,
        'month': month,
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '03:00', '06:00',
            '09:00', '12:00', '15:00',
            '18:00', '21:00',
        ],
    },
    file)


def retrieve_upper_level(c, year: str, month: str, dir: str):
    print(f"Retreiving upper levels for {year}-{month}")

    file = os.path.join(dir, "upper.nc4")

    if os.path.exists(file):
        print(f"File {file} already exists, skipping")
        return

    c.retrieve(
        'reanalysis-cerra-pressure-levels',
        {
            'format': 'netcdf',
            'variable': [
                'geopotential',
            ],
            'pressure_level': [
                '50',
            ],
            'data_type': 'reanalysis',
            'product_type': 'analysis',
            'year': year,
            'month': month,
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '03:00', '06:00',
                '09:00', '12:00', '15:00',
                '18:00', '21:00',
            ],
        },
        file)

def retrieve_lower_level(c, year: str, month: str, dir: str):
    print(f"Retreiving lower levels for {year}-{month}")

    file = os.path.join(dir, "lower.nc4")

    if os.path.exists(file):
        print(f"File {file} already exists, skipping")
        return

    c.retrieve(
        'reanalysis-cerra-pressure-levels',
        {
            'format': 'netcdf',
            'variable': [
                'geopotential', 'u_component_of_wind', 'v_component_of_wind',
            ],
            'pressure_level': [
                '1000',
            ],
            'data_type': 'reanalysis',
            'product_type': 'analysis',
            'year': year,
            'month': month,
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '03:00', '06:00',
                '09:00', '12:00', '15:00',
                '18:00', '21:00',
            ],
        },
        file)

def retrieve_surface(c, year: str, month: str, dir: str):
    print(f"Retreiving surface levels for {year}-{month}")

    file = os.path.join(dir, "surface.nc4")

    if os.path.exists(file):
        print(f"File {file} already exists, skipping")
        return


    c.retrieve(
        'reanalysis-cerra-single-levels',
        {
            'format': 'netcdf',
            'variable': [
                '10m_wind_direction', '10m_wind_speed', '2m_temperature',
                'mean_sea_level_pressure', 'surface_pressure', 'total_column_integrated_water_vapour',
            ],
            'level_type': 'surface_or_atmosphere',
            'data_type': 'reanalysis',
            'product_type': 'analysis',
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'month': month,
            'year': year,
            'time': [
                '00:00', '03:00', '06:00',
                '09:00', '12:00', '15:00',
                '18:00', '21:00',
            ],
        },
        file)
    
def retrieve_t2m(c, year: str, month: str, dir: str):
    print(f"Retreiving t2m for {year}-{month}")

    file = os.path.join(dir, "t2m.nc4")

    if os.path.exists(file):
        print(f"File {file} already exists, skipping")
        return


    c.retrieve(
        'reanalysis-cerra-single-levels',
        {
            'format': 'netcdf',
            'variable':   '2m_temperature',
            'level_type': 'surface_or_atmosphere',
            'data_type': 'reanalysis',
            'product_type': 'analysis',
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'month': month,
            'year': year,
            'time': [
                '00:00', '03:00', '06:00',
                '09:00', '12:00', '15:00',
                '18:00', '21:00',
            ],
        },
        file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/CERRA/")
    parser.add_argument("--start_year", type=int, default=2016)
    parser.add_argument("--end_year", type=int, default=2019)
    args = parser.parse_args()
    main(args.output_dir, args.start_year, args.end_year)