import cdsapi
import os
import argparse


area_cerra = [80, -60, 15, 80]
area_full = [90, -180, -90, 180]


def main(output_dir, start_year: int = 2016, end_year: int = 2019, full: bool = False):

    client = cdsapi.Client()

    if full:
        area = area_full
        print("Downloading ERA5 for the entire globe")
    else:
        area = area_cerra
        print("Downloading ERA5 for the CERRA region")

    for year in range(start_year,end_year+1):
        for month in range(1,13):

            month_str = f"{month:02d}"
            year_str = str(year)

            dir = os.path.join(output_dir, year_str, month_str)
            if not os.path.exists(dir):
                os.makedirs(dir)

            retreieve_upper_level(client, year_str, month_str, dir, area)
            retreieve_middle_levels(client, year_str, month_str, dir, area)
            retreieve_lower_level(client, year_str, month_str, dir, area)
            retreieve_surface(client, year_str, month_str, dir, area)


def retreieve_middle_levels(c, year: str, month: str, dir: str, area):
    print(f"Retreiving middle levels for {year}-{month}")

    file = os.path.join(dir, "middle.grib")
    
    if os.path.exists(file):
        print(f"File {file} already exists, skipping")
        return

    c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'format': 'grib',
        'variable': [
            'geopotential', 'relative_humidity', 'temperature',
            'u_component_of_wind', 'v_component_of_wind',
        ],
        'pressure_level': [
            '500', '850',
        ],
        'product_type': 'reanalysis',
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
        'area': area,
    },
    file)


def retreieve_upper_level(c, year: str, month: str, dir: str, area):
    print(f"Retreiving upper levels for {year}-{month}")

    file = os.path.join(dir, "upper.grib")
    
    if os.path.exists(file):
        print(f"File {file} already exists, skipping")
        return


    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'format': 'grib',
            'variable': [
                'geopotential',
            ],
            'pressure_level': [
                '50',
            ],
            'product_type': 'reanalysis',
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
            'area': area,
        },
        file)

def retreieve_lower_level(c, year: str, month: str, dir: str, area):
    print(f"Retreiving lower levels for {year}-{month}")

    file = os.path.join(dir, "lower.grib")
    
    if os.path.exists(file):
        print(f"File {file} already exists, skipping")
        return


    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'format': 'grib',
            'variable': [
                'geopotential', 'u_component_of_wind', 'v_component_of_wind',
            ],
            'pressure_level': [
                '1000',
            ],
            'product_type': 'reanalysis',
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
            'area':area,
        },
        file)

def retreieve_surface(c, year: str, month: str, dir: str, area):
    print(f"Retreiving surface levels for {year}-{month}")

    file = os.path.join(dir, "surface.grib")
    
    if os.path.exists(file):
        print(f"File {file} already exists, skipping")
        return


    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'format': 'grib',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                'mean_sea_level_pressure', 'surface_pressure', 'total_column_water_vapour',
            ],
            'product_type': 'reanalysis',
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
            'area': area,
        },
        file)

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/ERA5/")
    parser.add_argument("--start_year", type=int, default=2016)
    parser.add_argument("--end_year", type=int, default=2017)
    parser.add_argument("--full", action="store_true", default=False)
    args = parser.parse_args()
    main(args.output_dir, args.start_year, args.end_year, args.full)