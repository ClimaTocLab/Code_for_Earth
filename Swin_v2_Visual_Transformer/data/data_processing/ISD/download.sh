for year in $(seq 2010 2021);
do
        wget -np -nH https://www.ncei.noaa.gov/data/global-hourly/archive/csv/$year.tar.gz
done