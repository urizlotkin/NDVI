import pandas as pd
import vaex
import gc
import psutil
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import pyarrow as pa


def print_ram_usage():
    """
    Prints the current RAM usage of the process and total available RAM.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    ram_usage = memory_info.rss / (1024 * 1024)  # MB
    total_memory = psutil.virtual_memory().total / (1024 * 1024)
    print(f"Current RAM usage: {ram_usage:.2f} MB")
    print(f"Total RAM available: {total_memory:.2f} MB")


def filter_temp_rain_by_dates(df, temp_rain):
    """
    Filters the large temp_rain Vaex DataFrame down
    to only rows matching sentinel's 'Date' column.
    """
    # Convert df to Pandas DataFrame
    df_pd = df.to_pandas_df()
    # sentinel['Date'] is presumably datetime or string
    unique_dates = df_pd['Date'].unique()
    return temp_rain[temp_rain['Date'].isin(unique_dates)]

def merge_in_chunks(df_vaex, temp_rain_vaex, chunk_size=10_000):
    """
    Merge two Vaex DataFrames in chunks based on a single composite `key`.
    
    Parameters
    ----------
    df_vaex : vaex.DataFrame
        A Vaex DataFrame you want to merge ("left" side).
    temp_rain_vaex : vaex.DataFrame
        Another Vaex DataFrame ("right" side) that must be joined on `key`.
        Typically, you'd have created the 'key' column in `temp_rain_vaex`
        beforehand or you can create it inside this function if needed.
    chunk_size : int
        Number of rows to process in each chunk from df_vaex.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the merged result.
    """
    merged_chunks = []
    temp_rain_vaex['key'] = (
        temp_rain_vaex['POINT_X'].astype('str') +
        '_' +
        temp_rain_vaex['POINT_Y'].astype('str') +
        '_' +
        temp_rain_vaex['Date'].astype('str')
    )
    
    print(f'''\n\n{temp_rain_vaex['key'][:5]}\n\n''')
    # Iterate in chunks over the 'left' Vaex DataFrame
    for start in range(0, df_vaex.shape[0], chunk_size):
        end = start + chunk_size
        # Extract a chunk (still Vaex)
        chunk_vaex = df_vaex[start:end]


        # 2) Create the composite key in the chunk
        #    (POINT_X, POINT_Y, Date must be convertible to string)
        chunk_vaex['key'] = (
            chunk_vaex['POINT_X'].astype('str') +
            '_' +
            chunk_vaex['POINT_Y'].astype('str') +
            '_' +
            chunk_vaex['Date'].astype('str')
        )


        # Perform the join on the 'key' (single column)
        merged_vaex = chunk_vaex.join(
            temp_rain_vaex,
            on='key',
            how='left',
            lsuffix='_left',
            rsuffix='_right'
        )

        # Convert joined chunk to Pandas and store in a list
        merged_chunk_pd = merged_vaex.to_pandas_df()
        merged_chunks.append(merged_chunk_pd)

        # Cleanup
        del merged_vaex, merged_chunk_pd
        gc.collect()

    # Concatenate all chunk results into one Pandas DataFrame
    merged_df = pd.concat(merged_chunks, ignore_index=True)

    # Drop or rename any unwanted columns
    if 'key' in merged_df.columns:
        merged_df.drop(columns=['key'], inplace=True, errors='ignore')
    if 'Date_right' in merged_df.columns:
        merged_df.drop(columns=['Date_right'], inplace=True, errors='ignore')
    if 'Date_left' in merged_df.columns:
        merged_df.rename(columns={'Date_left': 'Date'}, inplace=True)

    return merged_df


def filter_dataset_by_dates_and_write_it(input_path, output_path, date_column):
    """
    Filters an Arrow dataset based on unique dates in a specified column.

    Parameters
    ----------
    input_path : str
        Path to the input dataset file.
    output_path : str
        Path to save the filtered dataset.
    date_column : str
        Name of the column containing dates.

    Returns
    -------
    None
    """

    # Read sentinel data
    sentinel = pd.read_csv(input_path)
    unique_dates = set(sentinel[date_column].unique())

    # Create an Arrow Dataset
    dataset = ds.dataset('data/temp_rain_points.parquet', format="parquet")

    # Filter the dataset in streaming fashion
    with pq.ParquetWriter(output_path, schema=dataset.schema) as writer:
        for batch in dataset.to_batches():
            # batch is a RecordBatch
            batch_df = batch.to_pandas()

            # Filter in Pandas
            mask = batch_df[date_column].isin(unique_dates)
            filtered_df = batch_df[mask]

            if not filtered_df.empty:
                # Convert back to Arrow Table
                table = pa.Table.from_pandas(filtered_df, schema=dataset.schema)
                writer.write_table(table)

# ---------------------------------------------------
#                   MAIN SCRIPT
# ---------------------------------------------------
# filter_dataset_by_dates_and_write_it('data/final_data/modis_with_temp_rain.csv', 'data/temp_rain_modis_filtered.parquet', 'Date')


landset = pd.read_csv('data/modis_NDVI_2000-2025.csv')
print(landset.shape)
landset["Date"] = pd.to_datetime(landset["Date"], errors="coerce")
threshold_date = pd.Timestamp("2000-02-18")
filtered_df = landset[landset["Date"] >= threshold_date]
print(filtered_df.shape)
print(filtered_df.columns)
# print(filtered_df["POINT_X"].min())

temp_rain_landsat_filtered = pd.read_parquet('data/temp_rain_modis_filtered.parquet')
temp_rain_landsat_filtered['POINT_X'] = temp_rain_landsat_filtered['POINT_X'].round(3)
temp_rain_landsat_filtered['POINT_Y'] = temp_rain_landsat_filtered['POINT_Y'].round(3)
print(temp_rain_landsat_filtered.shape)
print(temp_rain_landsat_filtered.columns)
# print(temp_rain_landsat_filtered["POINT_X"].min())

merged_df = pd.merge(
    filtered_df,
    temp_rain_landsat_filtered,
    on=['POINT_X', 'POINT_Y', 'Date'],
    how='inner'
)
print(merged_df.shape)
print(merged_df.columns)
merged_df.drop(columns=['gridcode_y'], inplace=True)
merged_df = merged_df.rename(columns={'gridcode_x': 'gridcode'})
print(merged_df.shape)
print(merged_df.columns)

merged_df.to_csv("data/final_data/modis_with_temp_rain.csv", index=False)
