import pandas as pd
import polars as pl
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle



## Load Data
### Landsat
landsat = pd.read_csv('data/final_data/landsat_with_temp_rain.csv')
print(landsat.columns)
print(landsat.shape)
landsat.isnull().sum()
landsat = landsat.dropna(subset=['TX(homog)', 'TN(homog)'])
print(landsat.shape)
### Modis
modis = pd.read_csv('data/final_data/modis_with_temp_rain.csv')
# Get the list of columns
columns = list(modis.columns)

# Swap the positions of 'NDVI' and 'Satellite'
ndvi_index = columns.index('NDVI')
satellite_index = columns.index('Satellite')

columns[ndvi_index], columns[satellite_index] = columns[satellite_index], columns[ndvi_index]

# Reorder the DataFrame
modis = modis[columns]
print(modis.columns)
print(modis.shape)
modis.isnull().sum()
modis = modis.dropna(subset=['TX(homog)', 'TN(homog)', 'NDVI'])
modis['NDVI'] = modis['NDVI'] * 0.0001
print(modis.shape)
### Sentinel
sentinel = pd.read_csv('data/final_data/sentinel_with_temp_rain.csv')
print(sentinel.columns)
print(sentinel.shape)
sentinel.isnull().sum()
sentinel = sentinel.dropna(subset=['TX(homog)', 'TN(homog)'])
print(sentinel.shape)
### Marge all datasets
# Convert pandas DataFrames to Polars
landsat_pl = pl.from_pandas(landsat)
modis_pl = pl.from_pandas(modis)
sentinel_pl = pl.from_pandas(sentinel)

del landsat, modis, sentinel
gc.collect()
# Concatenate them efficiently
all_data = pl.concat([landsat_pl, modis_pl, sentinel_pl])

print(all_data.columns)
print(all_data.shape)


del landsat_pl, modis_pl, sentinel_pl
gc.collect()
all_data = all_data.drop(['point_long', 'point_lat', 'station', 'POINT_X', 'POINT_Y', 'FID', 'Precipitation station', 'Temp station', 'Satellite'])
print(all_data.shape)
all_data = all_data.with_columns(
    pl.col("Date").str.to_date().alias("Date")  # Ensure it's a Date type
)
all_data = all_data.with_columns(
    pl.col("Date").dt.ordinal_day().alias("day_of_year")
)
grid_1 = all_data.filter(all_data['gridcode'] == 1)
# grid_1 = grid_1.to_pandas()
grid_2 = all_data.filter(all_data['gridcode'] == 2)
# grid_2 = grid_2.to_pandas()
grid_3 = all_data.filter(all_data['gridcode'] == 3)
# grid_3 = grid_3.to_pandas()
grid_4 = all_data.filter(all_data['gridcode'] == 4)
# grid_4 = grid_4.to_pandas()


del all_data
gc.collect()
print(grid_1.shape, grid_2.shape, grid_3.shape, grid_4.shape)
print(grid_1.columns)


## Data pre-process
### Get information for each column
grid_1 = grid_1.to_pandas()
# altitude of the station
grid_1['Israel_30m'].describe()
# Slope
grid_1['Slope_Isra'].describe()
grid_1['Aspect_Isr'].describe()
# The type of soil
grid_1['HK_General'].describe()
grid_1['HK_General'].unique()
# contiounous value for gridcode from 0.8-4.99
grid_1['Aridity_in'].describe()
# type of vegetation
grid_1['Prime_unit'].describe()
grid_1['Prime_unit'].unique()
# What is the order of the closest stream from 0-9
grid_1['STRM_ORD_1'].describe()
# max temprature of day
grid_1['TX(homog)'].describe()
# min temprature of day
grid_1['TN(homog)'].describe()
# Rain in mm in this day
grid_1['mm'].describe()
### Handle missing values
#### GridCode 1
grid_1.isnull().sum()
grid_1['mm'] = grid_1['mm'].fillna(0)
#### GridCode 2
null_counts = grid_2.select(pl.all().is_null().sum())
print(null_counts)
grid_2 = grid_2.with_columns(
    grid_2["mm"].fill_null(0)
)
#### GridCode 3
null_counts = grid_3.select(pl.all().is_null().sum())
print(null_counts)
grid_3 = grid_3.with_columns(
    grid_3["mm"].fill_null(0)
)
#### GridCode 4
null_counts = grid_4.select(pl.all().is_null().sum())
print(null_counts)
grid_4 = grid_4.with_columns(
    grid_4["mm"].fill_null(0)
)
### Handle categorial features - Prime_unit, HK_General
#### GirdCode 1
grid_1['Prime_unit'].unique()
# Create a LabelEncoder instance
le = LabelEncoder()

# Fit and transform the column
grid_1['Prime_unit'] = le.fit_transform(grid_1['Prime_unit'])

# Check the unique encoded values
print(grid_1['Prime_unit'].unique())
grid_1['HK_General'].unique()
# Create a LabelEncoder instance
le = LabelEncoder()

# Fit and transform the column
grid_1['HK_General'] = le.fit_transform(grid_1['HK_General'])

# Check the unique encoded values
print(grid_1['HK_General'].unique())
#### GridCode 2
grid_2.select("HK_General").unique()
# Convert categorical column to numerical encoding
grid_2 = grid_2.with_columns(
    grid_2["HK_General"].cast(pl.Categorical).to_physical()
)

# Check the unique encoded values
print(grid_2["HK_General"].unique())
grid_2.select("Prime_unit").unique()
# Convert categorical column to numerical encoding
grid_2 = grid_2.with_columns(
    grid_2["Prime_unit"].cast(pl.Categorical).to_physical()
)

# Check the unique encoded values
print(grid_2["Prime_unit"].unique())
#### GridCode 3
grid_3.select("HK_General").unique()
# Convert categorical column to numerical encoding
grid_3 = grid_3.with_columns(
    grid_3["HK_General"].cast(pl.Categorical).to_physical()
)

# Check the unique encoded values
print(grid_3["HK_General"].unique())
grid_3.select("Prime_unit").unique()
# Convert categorical column to numerical encoding
grid_3 = grid_3.with_columns(
    grid_3["Prime_unit"].cast(pl.Categorical).to_physical()
)

# Check the unique encoded values
print(grid_3["Prime_unit"].unique())
#### GridCode 4
grid_4.select("HK_General").unique()
# Convert categorical column to numerical encoding
grid_4 = grid_4.with_columns(
    grid_4["HK_General"].cast(pl.Categorical).to_physical()
)

# Check the unique encoded values
print(grid_4["HK_General"].unique())
grid_4.select("Prime_unit").unique()
grid_4 = grid_4.drop(['Prime_unit'])
### Normilazed continious numric values
#### GridCode 1
continuous_cols = [
    col for col in grid_1.select_dtypes(include=['number']).columns
    if grid_1[col].nunique() > 10  # Adjust threshold if needed
]

print(continuous_cols)
continuous_cols = continuous_cols[1:]
if 'day_of_year' in continuous_cols:
    continuous_cols.remove('day_of_year')
# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform only continuous numeric columns
grid_1[continuous_cols] = scaler.fit_transform(grid_1[continuous_cols])
#### GirdCode 2
# Select numeric columns
numeric_cols = [col for col in grid_2.columns if grid_2[col].dtype in [pl.Int64, pl.Float64]]

# Filter continuous columns (more than 10 unique values)
continuous_cols = [col for col in numeric_cols if grid_2[col].n_unique() > 10]

print(continuous_cols)
# Remove the first column from the list (if needed)
continuous_cols = continuous_cols[1:]

# Apply Z-score normalization (mean=0, std=1) using Polars
grid_2 = grid_2.with_columns([
    ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
    for col in continuous_cols
])
#### GridCode 3
# Select numeric columns
numeric_cols = [col for col in grid_3.columns if grid_3[col].dtype in [pl.Int64, pl.Float64]]

# Filter continuous columns (more than 10 unique values)
continuous_cols = [col for col in numeric_cols if grid_3[col].n_unique() > 10]

print(continuous_cols)
# Remove the first column from the list (if needed)
continuous_cols = continuous_cols[1:]

# Apply Z-score normalization (mean=0, std=1) using Polars
grid_3 = grid_3.with_columns([
    ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
    for col in continuous_cols
])
#### GridCode 4
# Select numeric columns
numeric_cols = [col for col in grid_4.columns if grid_4[col].dtype in [pl.Int64, pl.Float64]]

# Filter continuous columns (more than 10 unique values)
continuous_cols = [col for col in numeric_cols if grid_4[col].n_unique() > 10]

print(continuous_cols)
# Remove the first column from the list (if needed)
continuous_cols = continuous_cols[1:]

# Apply Z-score normalization (mean=0, std=1) using Polars
grid_4 = grid_4.with_columns([
    ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
    for col in continuous_cols
])
grid_1 = pl.from_pandas(grid_1)


def compute_feature_importance_shap(polar_df, target_feature="NDVI"):
    """
    Computes and visualizes feature importance for predicting the target feature (NDVI)
    using SHAP values with an XGBoost model.
    
    Args:
        polar_df (pl.DataFrame): A Polars DataFrame containing features and the target.
        target_feature (str): The name of the target column.

    Returns:
        None (displays a bar chart of SHAP feature importance)
    """
    polar_df = polar_df.drop(['gridcode', 'Date', 'day_of_year', 'Aridity_in'])

    # Convert Polars DataFrame to Pandas
    pandas_df = polar_df.to_pandas()

    # Separate features and target
    X = pandas_df.drop(columns=[target_feature])  # All features except target
    y = pandas_df[target_feature]  # Target feature

    # Ensure there are no missing values
    X = X.fillna(0)
    y = y.fillna(0)

    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define and train XGBoost model
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    assert X_train.isna().sum().sum() == 0, "Missing values in training data"

    # Compute SHAP values
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train, check_additivity=False)
    return shap_values, X_train


shap_values_1, X_train_1 = compute_feature_importance_shap(grid_1, target_feature="NDVI")
# Save SHAP values
with open("shap_values_1.pkl", "wb") as f:
    pickle.dump(shap_values_1, f)

# Save X_train data
with open("X_train_1.pkl", "wb") as f:
    pickle.dump(X_train_1, f)
shap_values_2, X_train_2 = compute_feature_importance_shap(grid_2, target_feature="NDVI")
# Save SHAP values
with open("shap_values_2.pkl", "wb") as f:
    pickle.dump(shap_values_2, f)

# Save X_train data
with open("X_train_2.pkl", "wb") as f:
    pickle.dump(X_train_2, f)
shap_values_3, X_train_3 = compute_feature_importance_shap(grid_3, target_feature="NDVI")
# Save SHAP values
with open("shap_values_3.pkl", "wb") as f:
    pickle.dump(shap_values_3, f)

# Save X_train data
with open("X_train_3.pkl", "wb") as f:
    pickle.dump(X_train_3, f)
shap_values_4, X_train_4 = compute_feature_importance_shap(grid_4, target_feature="NDVI")
# Save SHAP values
with open("shap_values_4.pkl", "wb") as f:
    pickle.dump(shap_values_4, f)

# Save X_train data
with open("X_train_4.pkl", "wb") as f:
    pickle.dump(X_train_4, f)

