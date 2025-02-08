import pandas as pd
import vaex


# -------------------- Read data points --------------------
# Shape: (10,066, 15)
# Columns: 'OID_', 'Israel_30m', 'Slope_Isra', 'Aspect_Isr', 'HK_General',
#        'Aridity_in', 'Prime_unit', 'gridcode', 'STRM_ORD_1', 'mannged',
#        'POINT_X', 'POINT_Y', 'Station_na', 'station', 'number'

df = pd.read_csv('data/data points.csv')
# print(df.shape)
# print(df.columns)
# print(df.OID_.max())
print(df.isnull().sum())

# -------------------- MODIS 2000-2022 --------------------
# Shape: (5,154,450, 18)
# Columns: 'Date', 'DayOfYear', 'DetailedQA', 'EVI', 'NDVI', 'RelativeAzimuth',
#        'SolarZenith', 'SummaryQA', 'ViewZenith', 'latitude', 'longitude',
#        'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b07',
#        'point_long', 'point_lat', 'point_idx'
# Dates: 2000-02-18 -> 2022-12-03

ndvi_df = pd.read_parquet('data/NDVI.parquet')
# print(ndvi_df.shape)
# print(ndvi_df.columns)
# print(ndvi_df.iloc[0])
# print(len(ndvi_df.Date.unique()))
# print(ndvi_df.point_long[0])
# print(ndvi_df.point_lat[0])
ndvi_df.to_csv('data/final_data/modis_with_temp_rain.csv', index=False)

# -------------------- Read temp rain points --------------------
# Shape: (81,783,940, 17)
# Columns:
# 'Date',
# 'FID',
# 'Israel_30m' - hight , 
# 'Slope_Isra',
# 'Aspect_Isr',
# סוג קרקע 'HK_General',
# contiounous value for gridcode  'Aridity_in',
# סוגי צומח 'Prime_unit',
# 'gridcode',
# גודל הנחל הקרוב'STRM_ORD_1',
# 'POINT_X',
# 'POINT_Y',
# 'Precipitation station',
# 'Temp station',
# max temprature 'TX(homog)',
# min temrature 'TN(homog)', 
# how much rain this day 'mm'
# Dates: 2000-02-14 -> 2022-12-04
# temp_rain = pd.read_parquet('data/temp_rain_points.parquet')
# temp_rain = vaex.open('data/temp_rain_points.parquet')
# print(temp_rain.head(5))
# print(temp_rain.columns)
# Perform value counts
# fid_counts = temp_rain['FID'].value_counts()
# print(fid_counts)
# Find the maximum value in the 'FID' column
# print(len(temp_rain['Date'].unique()))

# print(f"The maximum value in 'FID' is: {max_fid}")


# -------------------- Landsat 1982-2024 --------------------
# Shape(1,645,267, 5)
# The smallest date is: 1982-10-23
# The largest date is: 2024-12-31
# df = pd.read_csv('data/landsat_data.csv')
# print(df.shape)
# print(df.columns)
# # Find the smallest and largest date
# smallest_date = df['Date'].min()
# largest_date = df['Date'].max()

# print(f"The smallest date is: {smallest_date}")
# print(f"The largest date is: {largest_date}")


# -------------------- Sentinel 2017-2024 --------------------
# Shape(1,085,152, 5)
# The smallest date is: 2017-03-28
# The largest date is: 2024-12-31
# df = pd.read_csv('data/sentinel_data.csv')
# print(df.shape)
# print(df.columns)