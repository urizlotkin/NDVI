import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import math
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from scipy.interpolate import make_interp_spline



def get_all_cordenates(df):
    unique_combinations = df.groupby(['POINT_X', 'POINT_Y']).size().reset_index().rename(columns={0: 'Count'})
    years_per_combination = {}
    df['Date'] = pd.to_datetime(df['Date'])
    for index, row in unique_combinations.iterrows():
        point_x = row['POINT_X']
        point_y = row['POINT_Y']
        years = df[(df['POINT_X'] == point_x) & (df['POINT_Y'] == point_y)]['Date'].dt.year.unique()
        years_per_combination[(point_x, point_y)] = years
    return years_per_combination


def get_coordinates(station_name, df):
    all_coordinates = []
    station_data = df[df['station'] == station_name]
    if len(station_data) > 0:
        for index, row in station_data.iterrows():
            all_coordinates.append((row['POINT_X'], row['POINT_Y']))
        return list(set(all_coordinates))
    else:
        return None
    

def plot_ndvi_for_year(coordenate, year, df):
    """
    Plots NDVI values for a specific year in a timeline of months.
    
    Parameters:
        year (int): The year to plot.
        df (pd.DataFrame): DataFrame containing 'Date' and 'NDVI' columns.
        
    Returns:
        None
    """
    df = df[(df['POINT_X'] == coordenate[0]) & (df['POINT_Y'] == coordenate[1])]
    df['Date'] = pd.to_datetime(df['Date'])
    df_year = df[df['Date'].dt.year == year]
    if df_year.empty:
        print(f"No data available for the year {year}.")
        return
    # Extract month for the x-axis
    df_year['Month'] = df_year['Date'].dt.month
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_year, x='Month', y='NDVI', marker='o', label=f"NDVI {year}")
    
    plt.title(f"NDVI Values Timeline for {year}", fontsize=16)
    plt.xlabel("Months", fontsize=14)
    plt.ylabel("NDVI", fontsize=14)
    plt.xticks(ticks=range(1, 13), labels=month_labels, fontsize=12)
    plt.tight_layout()
    
    plt.legend()
    plt.show()


def plot_avg_ndvi_by_month(df, gridcode):
    """
    Plots the average NDVI values per month for a specific gridcode.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'Date', 'gridcode', and 'NDVI' columns.
        gridcode (int): The gridcode to filter the data.
        
    Returns:
        None
    """
    # Ensure the Date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter the DataFrame by gridcode
    df_filtered = df[df['gridcode'] == gridcode]
    
    if df_filtered.empty:
        print(f"No data available for gridcode {gridcode}.")
        return
    
    # Extract month and calculate average NDVI per month
    df_filtered['Month'] = df_filtered['Date'].dt.month
    monthly_avg = df_filtered.groupby('Month')['NDVI'].mean().reset_index()
    
    # Month labels
    month_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep']
    monthly_avg['MonthLabel'] = monthly_avg['Month'].apply(lambda x: month_labels[x - 1])
    
    # Plot
    sns.set_style('whitegrid')
    sns.set_context('talk')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=monthly_avg, x='MonthLabel', y='NDVI', color='skyblue')
    
    # Beautify the chart
    plt.title(f"Average NDVI by Month for Gridcode {gridcode}", fontsize=16)
    plt.xlabel("Months", fontsize=14)
    plt.ylabel("Average NDVI", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    
    # Show plot
    plt.show()
    
    
def plot_monthly_ndvi_hist_by_6year_groups(df):
    """
    Plots grouped bar charts of average NDVI by month and year,
    in chunks of 6 years per subplot. Ensures each subplot
    has the months in the order Oct -> Sep on the x-axis.
    """

    # Ensure 'Date' is a datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Extract Month and Year
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # Create a month label for readability (Jan, Feb, etc.)
    df['MonthLabel'] = df['Month'].apply(lambda m: calendar.month_abbr[m] if pd.notnull(m) else None)

    # Find all unique years, sorted
    unique_years = sorted(df['Year'].dropna().unique())
    if len(unique_years) == 0:
        print("No valid years found in DataFrame.")
        return

    # Determine how many subplots we need: 1 subplot per chunk of up to 6 years
    n_subplots = math.ceil(len(unique_years) / 6)

    # Create the figure and axes
    fig, axes = plt.subplots(
        nrows=n_subplots,
        ncols=1,
        figsize=(12, 6 * n_subplots),
        sharex=False
    )

    # If there's only one subplot, 'axes' is not a list. Make it a list for uniform handling.
    if n_subplots == 1:
        axes = [axes]

    # Seaborn style
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Define the month order for consistent x-axis from Oct to Sep
    month_order = [calendar.month_abbr[m] for m in list(range(10, 13)) + list(range(1, 10))]

    for i in range(n_subplots):
        # Select up to 6 years for this subplot
        chunk_years = unique_years[i * 6 : (i + 1) * 6]
        chunk_df = df[df['Year'].isin(chunk_years)].copy()

        # Create grouped bar chart on the corresponding axes
        ax = axes[i]

        # Specify the 'order' parameter to fix month order from Oct to Sep
        sns.barplot(
            data=chunk_df,
            x="MonthLabel",
            y="NDVI",
            hue="Year",
            estimator=pd.Series.mean,
            ci=None,
            ax=ax,
            order=month_order  # Force x-axis to be Oct->Sep
        )

        ax.set_title(f"Average NDVI by Month (Years: {chunk_years[0]}–{chunk_years[-1]})")
        ax.set_xlabel("Month")
        ax.set_ylabel("Average NDVI")
        ax.legend(title="Year")

        # Rotate x labels if needed
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')

    plt.tight_layout()
    plt.show()





def plot_ndvi_4subplots_3months_each(df):
    """
    Plots average NDVI vs. Year, in 4 subplots,
    each subplot showing 3 consecutive months:
      1) Jan-Mar
      2) Apr-Jun
      3) Jul-Sep
      4) Oct-Dec
    
    Each line in a subplot represents one month, x-axis = Year, y-axis = mean NDVI.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
          - 'Date': datetime or string convertible to datetime
          - 'NDVI': numeric NDVI values
    """
    # 1) Ensure 'Date' is datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # 2) Extract Year and Month
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    # 3) Compute average NDVI by (Year, Month)
    monthly_avg = (
        df.groupby(['Year','Month'], as_index=False)['NDVI']
          .mean()
    )
    
    # 4) Pivot so each month is a column, index=Year
    pivot_df = monthly_avg.pivot(index='Year', columns='Month', values='NDVI').sort_index()
    
    # 5) Prepare 4 subplots in a 2x2 layout
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10), sharey=True)
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    # Define the 4 groups of months
    month_groups = [
        [1, 2, 3],   # Jan, Feb, Mar
        [4, 5, 6],   # Apr, May, Jun
        [7, 8, 9],   # Jul, Aug, Sep
        [10, 11, 12] # Oct, Nov, Dec
    ]
    # We'll also prepare short labels (Jan, Feb, etc.)
    month_abbr = [calendar.month_abbr[i] for i in range(1,13)]
    
    # Flatten axes for easier indexing: axes[0][0], axes[0][1], ...
    axes_flat = axes.flatten()

    for i, group in enumerate(month_groups):
        ax = axes_flat[i]
        
        # For each month in this group, plot a line
        for m in group:
            if m in pivot_df.columns:
                ax.plot(
                    pivot_df.index,   # x = Year
                    pivot_df[m],      # y = NDVI for month m
                    marker='o',
                    label=month_abbr[m-1]  # e.g. month_abbr[0] -> 'Jan'
                )
        
        # Title showing which months
        group_labels = [month_abbr[m-1] for m in group]
        ax.set_title(f"Months: {', '.join(group_labels)}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Average NDVI")
        ax.legend(title="Month", loc="best")

    plt.tight_layout()
    plt.show()
    

def convert_to_seasonal(ts):
    """
    Given a pandas Timestamp ts, convert it to a "seasonal" date.
    If ts.month is between October and December, assign year 2000.
    If ts.month is between January and September, assign year 2001.
    
    If the day is February 29, it is converted to February 28.
    """
    # Adjust leap day: if February 29, set day to 28.
    day = ts.day
    if ts.month == 2 and ts.day == 29:
        day = 28
        
    if ts.month >= 10:
        return pd.Timestamp(f"2000-{ts.month:02d}-{day:02d}")
    else:
        return pd.Timestamp(f"2001-{ts.month:02d}-{day:02d}")


def plot_ndvi_min_max_by_year(df):
    """
    Plots three charts based on the mean NDVI values computed per date across all grid points.
    For each date, the mean NDVI across all 'point_long' and 'point_lat' is computed.
    Then, for each year, the date with the maximum mean NDVI and the date with the minimum mean NDVI are selected.
    
    The function produces:
      1. A chart of yearly min and max mean NDVI values (x-axis: Year; y-axis: NDVI).
      2. A chart of the date corresponding to the maximum mean NDVI per year.
         The x-axis is years and the y-axis is a "seasonal" date (from October to September).
         Only years where the seasonal max date is between November and April are plotted.
      3. A chart of the date corresponding to the minimum mean NDVI per year.
         The x-axis is years and the y-axis is a fixed-year date in the standard calendar.
         Only years where the min date (fixed to year 2000) is between April and November are plotted,
         but the y-axis spans from January 1 to December 30.
    
    A table with Year, Max Mean NDVI, Max Date, Min Mean NDVI, and Min Date is printed.
    """
    # --- Step 1: Preprocessing ---
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'NDVI'])
    
    # --- Step 2: Compute daily mean NDVI across all grid points ---
    daily_means = df.groupby('Date', as_index=False)['NDVI'].mean()
    daily_means['Year'] = daily_means['Date'].dt.year
    
    # --- Step 3: For each Year, select the date with the maximum and minimum daily mean NDVI ---
    def get_yearly_stats(group):
        max_ndvi = group['NDVI'].max()
        min_ndvi = group['NDVI'].min()
        max_date = group.loc[group['NDVI'].idxmax(), 'Date']
        min_date = group.loc[group['NDVI'].idxmin(), 'Date']
        return pd.Series({
            'max_ndvi': max_ndvi,
            'min_ndvi': min_ndvi,
            'max_date': max_date,
            'min_date': min_date
        })
    
    yearly_stats = daily_means.groupby('Year').apply(get_yearly_stats).reset_index()
    
    # --- Plot 1: Yearly NDVI Min and Max Values ---
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_stats['Year'], yearly_stats['min_ndvi'], label='Min Mean NDVI', marker='o', color='blue')
    plt.plot(yearly_stats['Year'], yearly_stats['max_ndvi'], label='Max Mean NDVI', marker='o', color='red')
    plt.xlabel('Year')
    plt.ylabel('Mean NDVI')
    plt.title('Yearly Min and Max Mean NDVI Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    # --- Step 4: Convert Dates for Additional Plots ---
    # For the max plot, convert the date to a seasonal date.
    yearly_stats['max_date_fixed'] = yearly_stats['max_date'].apply(convert_to_seasonal)
    # For the min plot, convert the date to a fixed-year date in the standard calendar (year 2000)
    yearly_stats['min_date_fixed'] = pd.to_datetime(yearly_stats['min_date'].dt.strftime("2000-%m-%d"))
    
    # --- Step 5: Filter Data for Additional Plots ---
    # For the max plot: include only rows where the seasonal max date's month is between November and April.
    max_filter = yearly_stats['max_date_fixed'].dt.month.isin([11, 12]) | yearly_stats['max_date_fixed'].dt.month.isin([1,2,3,4])
    max_plot_data = yearly_stats[max_filter].copy()
    
    # For the min plot: include only rows where the fixed min date's month is between April and November.
    min_filter = (yearly_stats['min_date_fixed'].dt.month >= 4) & (yearly_stats['min_date_fixed'].dt.month <= 11)
    min_plot_data = yearly_stats[min_filter].copy()
    
    # --- Plot 2: Yearly Max NDVI Date (Seasonal Ordering: Oct to Sep) ---
    plt.figure(figsize=(10, 6))
    plt.plot(max_plot_data['Year'], max_plot_data['max_date_fixed'], marker='o', linestyle='-', color='red', label='Max NDVI Date')
    plt.xlabel("Year")
    plt.ylabel("Max NDVI Date (Month-Day)")
    plt.title("Yearly Date of Maximum Mean NDVI (Filtered: Nov-Apr)")
    ax = plt.gca()
    ax.yaxis.set_major_locator(mdates.MonthLocator())
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%b"))
    # For seasonal ordering: set y-axis limits from October 1, 2000 to September 30, 2001.
    ax.set_ylim(pd.Timestamp("2000-10-01"), pd.Timestamp("2001-09-30"))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # --- Plot 3: Yearly Min NDVI Date (Standard Ordering: Jan to Dec) ---
    plt.figure(figsize=(10, 6))
    plt.plot(min_plot_data['Year'], min_plot_data['min_date_fixed'], marker='o', linestyle='-', color='blue', label='Min NDVI Date')
    plt.xlabel("Year")
    plt.ylabel("Min NDVI Date (Month-Day)")
    plt.title("Yearly Date of Minimum Mean NDVI (Filtered: Apr-Nov, Y-axis: Jan-Dec)")
    ax = plt.gca()
    ax.yaxis.set_major_locator(mdates.MonthLocator())
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%b"))
    # For standard ordering, set y-axis limits from January 1, 2000 to December 30, 2000.
    ax.set_ylim(pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-30"))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
# df_pandas = pd.read_csv('your_ndvi_data.csv')  # Ensure the CSV has Date, NDVI, point_long, point_lat columns.
# plot_ndvi_min_max_by_year(df_pandas)




def plot_ndvi_max_min_month_by_year(df):
    """
    Creates two separate graphs:
    1. Max NDVI Month: X-axis as Years, Y-axis as Months (Oct -> Sep order).
    2. Min NDVI Month: X-axis as Years, Y-axis as Months (Apr -> Mar order).
    """
    # Ensure 'Date' is a datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Extract Year and Month
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Drop rows with missing values
    df = df.dropna(subset=['Date', 'NDVI'])

    # Group by Year and Month to calculate average NDVI
    monthly_avg_ndvi = df.groupby(['Year', 'Month'])['NDVI'].mean().reset_index()

    # Find the months with the max and min average NDVI for each year
    yearly_stats = monthly_avg_ndvi.groupby('Year').agg(
        max_month=('Month', lambda x: x[monthly_avg_ndvi.loc[x.index, 'NDVI'].idxmax()]),
        min_month=('Month', lambda x: x[monthly_avg_ndvi.loc[x.index, 'NDVI'].idxmin()])
    ).reset_index()

    # Adjust months for Oct -> Sep order (for max)
    max_month_order = list(range(10, 13)) + list(range(1, 10))
    max_month_mapping = {month: i for i, month in enumerate(max_month_order, start=1)}
    yearly_stats['max_month_mapped'] = yearly_stats['max_month'].map(max_month_mapping)

    # Adjust months for Apr -> Mar order (for min)
    min_month_order = list(range(4, 13)) + list(range(1, 4))
    min_month_mapping = {month: i for i, month in enumerate(min_month_order, start=1)}
    yearly_stats['min_month_mapped'] = yearly_stats['min_month'].map(min_month_mapping)

    # Plot for max NDVI months
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_stats['Year'], yearly_stats['max_month_mapped'], label='Max NDVI Month', marker='o', color='red')
    max_month_labels = [calendar.month_abbr[m] for m in max_month_order]
    plt.yticks(ticks=range(1, 13), labels=max_month_labels)
    plt.xlabel('Year')
    plt.ylabel('Month')
    plt.title('Yearly Max Average NDVI Months (Oct -> Sep)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot for min NDVI months
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_stats['Year'], yearly_stats['min_month_mapped'], label='Min NDVI Month', marker='o', color='blue')
    min_month_labels = [calendar.month_abbr[m] for m in min_month_order]
    plt.yticks(ticks=range(1, 13), labels=min_month_labels)
    plt.xlabel('Year')
    plt.ylabel('Month')
    plt.title('Yearly Min Average NDVI Months (Apr -> Mar)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def find_rainy_season_months(months):
    """
    Given a list of month numbers (1-12), finds:
    - The closest month after September (10, 11, 12, 1, 2, 3) -> start of rainy season
    - The closest month before September (8, 7, 6, 5, 4, 3, 2, 1) -> end of rainy season
    """
    # Define valid months for start and end of rainy season
    after_sept_options = [10, 11, 12, 1, 2, 3]
    before_sept_options = [8, 7, 6, 5, 4, 3, 2, 1]
    
    # Find closest month after September (preferably from 10, 11, 12 first)
    after_sept_months = [m for m in months if m in after_sept_options]
    if after_sept_months:
        start_rainy_season = min(after_sept_months, key=lambda x: (abs(9 - x)))
    else:
        start_rainy_season = None

    # Find the closest month before September
    before_sept_months = [m for m in months if m in before_sept_options]
    end_rainy_season = max(before_sept_months, default=None)

    return start_rainy_season, end_rainy_season


def plot_ndvi_rise_and_fall_intervals_one_graph(df, threshold=0.7):
    """
    Plots intervals for each 15-year bin. In each bin, years go from:
      bin_start ... bin_start+14
    A separate figure is produced for each 15-year range found in the data.
    
    (Modified to produce a SINGLE plot for all years instead.)
    """

    # Ensure 'Date' is datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Extract Year and Month
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Drop rows with missing NDVI or Date
    df = df.dropna(subset=['Date', 'NDVI'])

    # Filter out invalid months
    df = df[(df['Month'] >= 1) & (df['Month'] <= 12)]

    # Sort by Date
    df = df.sort_values(by='Date')

    # List to store intervals (Year, StartMonth, EndMonth)
    intervals = []

    # -- For each year, compute monthly-average NDVI, then find the months above threshold
    for year, year_data in df.groupby('Year'):
        monthly_data = year_data.groupby('Month', as_index=False)['NDVI'].mean()
        if monthly_data.empty:
            continue

        # Average NDVI across months in the year
        yearly_avg_ndvi = monthly_data['NDVI'].mean()

        # Example threshold: 100% of year's avg NDVI
        value = threshold * yearly_avg_ndvi  

        # Identify months whose average NDVI is above threshold
        above_threshold_months = monthly_data.loc[monthly_data['NDVI'] > value, 'Month'].sort_values()
        if len(above_threshold_months) > 0:
            months = above_threshold_months.values.tolist()
            start_month, end_month = find_rainy_season_months(months)
            intervals.append((year, start_month, end_month))

    # Convert intervals to a DataFrame
    interval_df = pd.DataFrame(intervals, columns=['Year', 'StartMonth', 'EndMonth'])

    # If no intervals, just return
    if interval_df.empty:
        print("No intervals found where monthly NDVI > threshold in any year.")
        return

    # Define the custom month order (Oct -> Sep) and map
    month_order = list(range(10, 13)) + list(range(1, 10))
    month_mapping = {month: i for i, month in enumerate(month_order, start=1)}

    # Filter out intervals with invalid months
    interval_df = interval_df[
        (interval_df['StartMonth'] >= 1) & (interval_df['StartMonth'] <= 12) &
        (interval_df['EndMonth']   >= 1) & (interval_df['EndMonth']   <= 12)
    ]
    if interval_df.empty:
        print("All intervals had invalid months.")
        return

    # Map months to the custom (Oct->Sep) scale
    interval_df['StartMonthMapped'] = interval_df['StartMonth'].map(month_mapping)
    interval_df['EndMonthMapped']   = interval_df['EndMonth'].map(month_mapping)

    # If the mapped end month is smaller than the start month, swap them
    swap_indices = interval_df['EndMonthMapped'] < interval_df['StartMonthMapped']
    interval_df.loc[swap_indices, ['StartMonthMapped', 'EndMonthMapped']] = \
        interval_df.loc[swap_indices, ['EndMonthMapped', 'StartMonthMapped']].values

    # ---------------------------------------------------
    #   GROUP BY 15-YEAR BINS (ORIGINALLY)
    #   NOW: PLOT ALL IN ONE CHART
    # ---------------------------------------------------
    min_year = interval_df['Year'].min()
    max_year = interval_df['Year'].max()

    # We still compute BinIndex for minimal code changes
    interval_df['BinIndex'] = (interval_df['Year'] - min_year) // 15

    # Print available 15-yr bins for debug
    print("15-year bin indexes found in data:", interval_df['BinIndex'].unique())

    # Instead of looping over bins, we do a single figure
    bin_df = interval_df
    bin_start_year = min_year  # earliest year
    bin_end_year   = max_year  # latest year

    plt.figure(figsize=(12, 10))  # one figure

    # Plot each row's interval
    first_label_used = False
    for _, row in bin_df.iterrows():
        label = 'NDVI > threshold' if not first_label_used else None
        plt.plot(
            [row['StartMonthMapped'], row['EndMonthMapped']],  # X-axis: start -> end
            [row['Year'], row['Year']],                       # Y-axis: that year
            marker='o', color='blue', label=label
        )
        if not first_label_used:
            first_label_used = True

    # Customize x-axis (Oct -> Sep)
    month_labels = [calendar.month_abbr[m] for m in month_order]
    plt.xticks(ticks=range(1, 13), labels=month_labels)

    # Labels, title, legend
    plt.xlabel('Month (Oct -> Sep)')
    plt.ylabel('Year')
    plt.title(f'NDVI Above Threshold — {bin_start_year}-{bin_end_year}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print intervals in this single chart for inspection
    print(f"Intervals for {bin_start_year}-{bin_end_year}:")
    print(bin_df[['Year','StartMonth','EndMonth']])
    print("-" * 60)


    # Print intervals in this single chart for inspection
    print(f"Intervals for {bin_start_year}-{bin_end_year}:")
    print(bin_df[['Year','StartMonth','EndMonth']])
    print("-" * 60)



def plot_ndvi_rise_and_fall_intervals_15yr(df, threshold=1.0):
    """
    Plots intervals for each 15-year bin. In each bin, years go from:
      bin_start ... bin_start+14
    A separate figure is produced for each 15-year range found in the data.
    """

    # Ensure 'Date' is datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Extract Year and Month
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Drop rows with missing NDVI or Date
    df = df.dropna(subset=['Date', 'NDVI'])

    # Filter out invalid months
    df = df[(df['Month'] >= 1) & (df['Month'] <= 12)]

    # Sort by Date
    df = df.sort_values(by='Date')

    # List to store intervals (Year, StartMonth, EndMonth)
    intervals = []

    # -- For each year, compute monthly-average NDVI, then find the months above threshold
    for year, year_data in df.groupby('Year'):
        monthly_data = year_data.groupby('Month', as_index=False)['NDVI'].mean()
        if monthly_data.empty:
            continue

        # Average NDVI across months in the year
        yearly_avg_ndvi = monthly_data['NDVI'].mean()

        # Example threshold: 50% of year's avg NDVI
        value = threshold * yearly_avg_ndvi  

        # Identify months whose average NDVI is above threshold
        above_threshold_months = monthly_data.loc[monthly_data['NDVI'] > value, 'Month'].sort_values()
        if len(above_threshold_months) > 0:
            months = above_threshold_months.values.tolist()
            start_month, end_month = find_rainy_season_months(months)
            intervals.append((year, start_month, end_month))

    # Convert intervals to a DataFrame
    interval_df = pd.DataFrame(intervals, columns=['Year', 'StartMonth', 'EndMonth'])

    # If no intervals, just return
    if interval_df.empty:
        print("No intervals found where monthly NDVI > threshold in any year.")
        return

    # Define the custom month order (Oct -> Sep) and map
    month_order = list(range(10, 13)) + list(range(1, 10))
    month_mapping = {month: i for i, month in enumerate(month_order, start=1)}

    # Filter out intervals with invalid months
    interval_df = interval_df[
        (interval_df['StartMonth'] >= 1) & (interval_df['StartMonth'] <= 12) &
        (interval_df['EndMonth']   >= 1) & (interval_df['EndMonth']   <= 12)
    ]
    if interval_df.empty:
        print("All intervals had invalid months.")
        return

    # Map months to the custom (Oct->Sep) scale
    interval_df['StartMonthMapped'] = interval_df['StartMonth'].map(month_mapping)
    interval_df['EndMonthMapped']   = interval_df['EndMonth'].map(month_mapping)

    # If the mapped end month is smaller than the start month, swap them
    swap_indices = interval_df['EndMonthMapped'] < interval_df['StartMonthMapped']
    interval_df.loc[swap_indices, ['StartMonthMapped', 'EndMonthMapped']] = \
        interval_df.loc[swap_indices, ['EndMonthMapped', 'StartMonthMapped']].values

    # ---------------------------------------------------
    #   GROUP BY 15-YEAR BINS
    # ---------------------------------------------------
    # Find the earliest year in the data
    min_year = interval_df['Year'].min()

    # Create a bin index for each row. For example:
    #   bin_index = 0 means [min_year ... min_year+14]
    #   bin_index = 1 means [min_year+15 ... min_year+29], etc.
    interval_df['BinIndex'] = (interval_df['Year'] - min_year) // 15

    # Prepare month labels once
    month_labels = [calendar.month_abbr[m] for m in month_order]

    # Print available 15-yr bins for debug
    print("15-year bin indexes found in data:", interval_df['BinIndex'].unique())

    # Loop over each bin
    for bin_index, bin_df in interval_df.groupby('BinIndex'):
        # Calculate the start and end year for labeling
        bin_start_year = min_year + bin_index * 15
        bin_end_year   = bin_start_year + 14

        # Create a new figure for this 15-year bin
        plt.figure(figsize=(12, 6))

        # Plot each row's interval within this bin
        first_label_used = False
        for _, row in bin_df.iterrows():
            label = 'NDVI > threshold' if not first_label_used else None
            plt.plot(
                [row['StartMonthMapped'], row['EndMonthMapped']],  # X-axis: start -> end
                [row['Year'], row['Year']],                       # Y-axis: that year
                marker='o', color='blue', label=label
            )
            if not first_label_used:
                first_label_used = True

        # Customize x-axis (Oct -> Sep)
        plt.xticks(ticks=range(1, 13), labels=month_labels)

        # Labels, title, legend
        plt.xlabel('Month (Oct -> Sep)')
        plt.ylabel('Year')
        plt.title(f'NDVI Above Threshold — {bin_start_year}-{bin_end_year}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Show or save the figure
        plt.show()


def find_big_change_start_date(df):
    """
    For each year (>= 1985), this function:
      1. Computes the monthly average NDVI for that year.
      2. Calculates the month-on-month percentage change.
      3. Identifies the month with the largest positive percentage change.
      4. Then, within that month, computes the average NDVI for each unique date.
      5. Returns the date (with the lowest average NDVI) in that month.
         However, if this "best date" falls between June and September, that year is skipped.
    
    Returns:
      A pandas DataFrame with columns: Year and StartDate.
    """
    # Ensure the Date column is datetime and extract Year and Month.
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df = df.dropna(subset=['NDVI'])
    
    results = []
    for year, grp in df.groupby('Year'):
        # Skip years before 1985.
        if year < 1985:
            continue

        # 1) Compute monthly average NDVI for this year.
        monthly_avg = grp.groupby('Month')['NDVI'].mean().sort_index()
        if monthly_avg.empty or len(monthly_avg) < 2:
            results.append((year, pd.NaT))
            continue

        # 2) Compute the month-on-month percentage change.
        monthly_pct_change = monthly_avg.pct_change() * 100

        # 3) Find the month with the largest positive percentage change (ignoring the first month).
        largest_pct_change_month = monthly_pct_change.iloc[1:].idxmax()

        # 4) Within that month, compute the average NDVI for each unique date.
        selected_month_data = grp[grp['Month'] == largest_pct_change_month]
        date_means = selected_month_data.groupby('Date')['NDVI'].mean()

        # 5) Identify the date with the smallest (lowest) mean NDVI.
        best_date = date_means.idxmin()
        
        # If best_date falls between June and September, skip this year.
        if 2 <= best_date.month <= 9:
            continue

        results.append((year, best_date))
    
    result_df = pd.DataFrame(results, columns=['Year', 'StartMonth'])
    return result_df




def find_ndvi_start_date_3_consecutive(df):
    """
    For each year (>= 1985), find the first date where NDVI increases for 5 consecutive days,
    ensuring that each year's order starts from September 1st and ends in August 31st.
    
    Parameters:
      df (DataFrame): DataFrame containing at least 'Date' and 'NDVI' columns.
      
    Returns:
      DataFrame: A DataFrame with columns 'Year' and 'NDVI_Start_Date'.
    """
    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Extract year and month
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df = df.dropna(subset=['NDVI'])
    
    results = []

    for year, grp in df.groupby('Year'):
        # Skip years before 1985
        if year < 1985:
            continue
        
        # Create a custom sorting key that makes September the first month
        grp = grp.assign(
            CustomSortKey=np.where(grp['Month'] >= 9, grp['Month'] - 9, grp['Month'] + 3)
        )
        
        # Sort the group by this custom key and Date (for correct ordering within a month)
        grp = grp.sort_values(['CustomSortKey', 'Date']).reset_index(drop=True)

        start_date = None

        # We need at least 5 days to check for a 5-day increase.
        if len(grp) < 5:
            results.append((year, np.nan))
            continue

        # Iterate through the records and check for 5 consecutive increasing NDVI values.
        for i in range(len(grp) - 3):
            if (grp.loc[i + 1, 'NDVI'] > grp.loc[i, 'NDVI'] and
                grp.loc[i + 2, 'NDVI'] > grp.loc[i + 1, 'NDVI']):
                start_date = grp.loc[i + 1, 'Date']
                break
        if start_date == None or not (8 <= start_date.month <= 12):
            continue
        results.append((year, start_date))
    # Convert results to a DataFrame.
    start_date_df = pd.DataFrame(results, columns=['Year', 'StartMonth'])
    return start_date_df


def plot_start_date_across_years(df_start):
    """
    df_start: a DataFrame with columns ['Year','StartMonth'].
              'StartMonth' should be a date (YYYY-MM-DD).
    
    Plots a line connecting points of NDVI start-date across years,
    starting from 1985, with months ordered from September to August.
    """
    # Convert 'StartMonth' to datetime if not already
    df_start['StartMonth'] = pd.to_datetime(df_start['StartMonth'], errors='coerce')

    # Extract month with rounding: if day > 15, shift to next month
    df_start['Month'] = df_start['StartMonth'].dt.month + (df_start['StartMonth'].dt.day > 15).astype(int)
    
    # Ensure December rolls over to January correctly
    df_start['Month'] = df_start['Month'].apply(lambda x: 1 if x == 13 else x)

    # Remove rows with missing StartMonth
    df_start = df_start.dropna(subset=['StartMonth'])

    # Sort by Year so the line connects chronologically
    df_start = df_start.sort_values(by='Year')

    # Define custom month order: Aug (8) → Jul (7)
    custom_order = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7]
    month_to_rank = {m: i+1 for i, m in enumerate(custom_order)}

    # Create a new column with the "rank" (1 = August, 12 = July)
    df_start['MonthRank'] = df_start['Month'].map(month_to_rank)

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot line with markers
    plt.plot(
        df_start['Year'],
        df_start['MonthRank'],
        marker='o',
        color='blue',
        label='Start of NDVI Rise'
    )

    # Ensure X-axis starts at 1985
    plt.xlim(left=1985)

    # Invert the Y-axis so August (rank=1) is at the top
    plt.gca().invert_yaxis()

    # Label Y-axis with months in order [8..7]
    plt.yticks(
        ticks=range(1, 13),
        labels=[pd.to_datetime(f'2023-{m}-01').strftime('%B') for m in custom_order]  # Month names
    )

    # Axis labels, title, legend
    plt.xlabel("Year (skipping < 1985)")
    plt.ylabel("Month (Aug at Top → Jul at Bottom)")
    plt.title("Start Date of NDVI Rise Across Years")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_correlation_with_target(polar_df, target_feature="NDVI"):
    # Convert Polar DataFrame to Pandas
    polar_df = polar_df.drop(['gridcode'])
    pandas_df = polar_df.to_pandas()

    # Compute correlation matrix
    correlation_matrix = pandas_df.corr()

    # Extract correlation of all features with the target feature
    target_correlation = correlation_matrix[[target_feature]].drop(index=target_feature)

    # Sort correlations for better visualization
    target_correlation = target_correlation.sort_values(by=target_feature, ascending=False)

    # Plot using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(target_correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f"Correlation of Features with {target_feature}")
    plt.show()
    
    
def plot_correlation_matrix(polar_df):
    """
    Plots a correlation matrix heatmap for all features in a Polars DataFrame.

    Args:
        polar_df (pl.DataFrame): A Polars DataFrame with numerical features.
    """
    # Convert Polar DataFrame to Pandas
    polar_df = polar_df.drop(['gridcode'])
    pandas_df = polar_df.to_pandas()

    # Compute correlation matrix
    correlation_matrix = pandas_df.corr()

    # Plot using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.show()
    

# import polars as pl
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import xgboost as xgb
# from sklearn.model_selection import train_test_split

def compute_feature_importance_xgb(polar_df, target_feature="NDVI", is_drop=True):
    """
    Computes and visualizes feature importance for predicting the target feature (NDVI)
    using an XGBoost model.
    
    Args:
        polar_df (pl.DataFrame): A Polars DataFrame containing features and the target.
        target_feature (str): The name of the target column.

    Returns:
        None (displays a bar chart of feature importance)
    """
    # Convert Polars DataFrame to Pandas
    if is_drop:
        polar_df = polar_df.drop(['gridcode', 'Date', 'day_of_year'])

        # Convert Polars DataFrame to Pandas
        pandas_df = polar_df.to_pandas()
    else:
        pandas_df = polar_df

    # Separate features and target
    X = pandas_df.drop(columns=[target_feature])  # All features except target
    y = pandas_df[target_feature]  # Target feature

    # Ensure there are no missing values
    X = X.fillna(0)
    y = y.fillna(0)

    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define XGBoost model
    model = xgb.XGBRegressor(n_estimators=100, random_state=42, importance_type="gain")

    # Train the model
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x="Importance", y="Feature", palette="coolwarm")
    plt.xlabel("Feature Importance (Gain)")
    plt.ylabel("Features")
    plt.title(f"Feature Importance for Predicting {target_feature} (XGBoost)")
    plt.show()
    
    
def compute_feature_importance_shap(polar_df, target_feature="NDVI", is_drop=True):
    """
    Computes and visualizes feature importance for predicting the target feature (NDVI)
    using SHAP values with an XGBoost model.
    
    Args:
        polar_df (pl.DataFrame): A Polars DataFrame containing features and the target.
        target_feature (str): The name of the target column.

    Returns:
        None (displays a bar chart of SHAP feature importance)
    """
    if is_drop:
        polar_df = polar_df.drop(['gridcode', 'Date', 'day_of_year'])

        # Convert Polars DataFrame to Pandas
        pandas_df = polar_df.to_pandas()
    else:
        pandas_df = polar_df

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

    # Compute SHAP values
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)
    return shap_values, X_train



def aggregate_group(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Given a pandas DataFrame for one group (one point_long, one point_lat, one year),
    compute the aggregated fields:
      - Static fields: first value for Israel_30m, Slope_Isra, Aspect_Isr, HK_General,
        Aridity_in, Prime_unit, STRM_ORD_1.
      - Monthly averages for TX(homog) and TN(homog) for months: September (9), October (10),
        November (11), December (12) and January (1).
      - Monthly totals for mm (precipitation) for the same months.
      - NDVI_Onset_DOY: the day_of_year of the first date at which NDVI increases for 5
        consecutive days (if any), using a custom seasonal sort (from September to January).
      - big_change_day: the day of the year corresponding to the day (in the month with the
        largest month-on-month NDVI percentage increase) that has the lowest NDVI mean.
    """
    # Ensure Date is datetime.
    pdf["Date"] = pd.to_datetime(pdf["Date"], errors="coerce")
    
    # --- Custom sort: sort the group so that the order is from September 1 to January 31.
    # Create custom sort keys: assign January (month == 1) a value of 13 so it comes after December.
    pdf["custom_month"] = pdf["Date"].dt.month.apply(lambda m: m if m != 1 else 13)
    pdf["custom_day"] = pdf["Date"].dt.day
    pdf = pdf.sort_values(by=["custom_month", "custom_day"]).reset_index(drop=True)
    
    # Prepare output dictionary.
    res = {}
    res["point_long"] = pdf["point_long"].iloc[0]
    res["point_lat"]  = pdf["point_lat"].iloc[0]
    res["year"]       = pdf["year"].iloc[0]
    
    # Static fields (assumed invariant in the group).
    for col in ["Israel_30m", "Slope_Isra", "Aspect_Isr", 
                "HK_General", "Aridity_in", "Prime_unit", "STRM_ORD_1"]:
        res[col] = pdf[col].iloc[0]
    
    # Define the month labels for which we want to compute monthly aggregates.
    # (We expect data for September (9), October (10), November (11), December (12) and January (1).)
    month_labels = {9: "sep", 10: "oct", 11: "nov", 12: "dec", 1: "jan"}
    for m, label in month_labels.items():
        sub = pdf[pdf["month"] == m]
        res[f"avg_TX(homog)_{label}"] = sub["TX(homog)"].mean() if not sub.empty else np.nan
        res[f"avg_TN(homog)_{label}"] = sub["TN(homog)"].mean() if not sub.empty else np.nan
        res[f"total_mm_{label}"] = sub["mm"].sum() if not sub.empty else np.nan

    # Compute NDVI_Onset_DOY using a 5-consecutive-day rising rule.
    ndvi_onset = np.nan  # default value if no sequence is found
    if len(pdf) >= 5:
        for i in range(len(pdf) - 2):
            if (pdf.loc[i + 1, "NDVI"] > pdf.loc[i, "NDVI"] and
                pdf.loc[i + 2, "NDVI"] > pdf.loc[i + 1, "NDVI"]):
                ndvi_onset = pdf.loc[i, "day_of_year"]
                break
    res["NDVI_Onset_DOY"] = ndvi_onset

    # --- NEW FEATURE: big_change_day (as day of year)
    # 1. Compute monthly average NDVI using the original 'month' column.
    monthly_avg = pdf.groupby("month")["NDVI"].mean().sort_index()
    if len(monthly_avg) < 2:
        big_change = np.nan
    else:
        # 2. Compute month-on-month percentage change (as a percentage).
        monthly_pct_change = monthly_avg.pct_change() * 100
        # Ignore the first month (which has no previous month) and get the month with the largest increase.
        big_change_month = monthly_pct_change.iloc[1:].idxmax()  # returns the month (as integer)
        
        # 3. Within the records for that month, group by day-of-month and compute average NDVI.
        sub = pdf[pdf["month"] == big_change_month].copy()
        if sub.empty:
            big_change = np.nan
        else:
            sub["day_of_month"] = sub["Date"].dt.day
            daily_avg = sub.groupby("day_of_month")["NDVI"].mean()
            # Get the day (of month) with the lowest NDVI mean.
            lowest_day = daily_avg.idxmin()
            # Construct the corresponding date using the group's year and the chosen month and day.
            target_date = pd.Timestamp(year=res["year"], month=big_change_month, day=lowest_day)
            big_change = target_date.timetuple().tm_yday
    res["big_change_day"] = big_change

    # Return the result as a one-row pandas DataFrame.
    return pd.DataFrame([res])

def features_extructor(grid: pl.DataFrame) -> pl.DataFrame:
    """
    Process a Polars DataFrame with daily data for each grid cell.
    
    Expected columns in grid:
      - "Date": a date string (e.g., "YYYY-MM-DD") or datetime.
      - "NDVI", "TX(homog)", "TN(homog)", "mm"
      - (The column "day_of_year" will be computed below.)
      - Static fields: "Israel_30m", "Slope_Isra", "Aspect_Isr", "HK_General",
        "Aridity_in", "Prime_unit", "STRM_ORD_1"
      - "point_long", "point_lat"
    
    The function:
      1. Ensures the Date column is a datetime and computes "year", "month", and "day_of_year".
      2. Converts the Polars DataFrame to a pandas DataFrame.
      3. Groups by "point_long", "point_lat", and "year" and applies the custom aggregation.
      4. Converts the result back into a Polars DataFrame.
    
    Returns:
      A Polars DataFrame with one row per grid cell per year.
    """
    # Ensure grid is a Polars DataFrame.
    if not isinstance(grid, pl.DataFrame):
        grid = pl.DataFrame(grid)
    
    # Step 1: Convert Date to datetime and add date parts.
    grid = grid.with_columns([
        pl.col("Date").cast(pl.Datetime("ms")).alias("Date"),
        pl.col("Date").dt.year().alias("year"),
        pl.col("Date").dt.month().alias("month"),
        pl.col("Date").dt.ordinal_day().alias("day_of_year")
    ])
    
    # Convert the Polars DataFrame to pandas for grouping.
    grid_pd = grid.to_pandas()
    
    # Group by point_long, point_lat, and year; group_keys=False ensures the group keys
    # do not become part of the index.
    grouped_pd = grid_pd.groupby(["point_long", "point_lat", "year"], group_keys=False).apply(aggregate_group)
    
    # Convert the resulting pandas DataFrame back into a Polars DataFrame.
    result = pl.from_pandas(grouped_pd)
    
    # Optionally, reorder/select columns in the desired order.
    desired_cols = [
        "point_long", "point_lat", "year",
        "Israel_30m", "Slope_Isra", "Aspect_Isr", "HK_General",
        "Aridity_in", "Prime_unit", "STRM_ORD_1",
        "avg_TX(homog)_sep", "avg_TX(homog)_oct", "avg_TX(homog)_nov",
        "avg_TX(homog)_dec", "avg_TX(homog)_jan",
        "avg_TN(homog)_sep", "avg_TN(homog)_oct", "avg_TN(homog)_nov",
        "avg_TN(homog)_dec", "avg_TN(homog)_jan",
        "NDVI_Onset_DOY", "big_change_day",
        "total_mm_sep", "total_mm_oct", "total_mm_nov",
        "total_mm_dec", "total_mm_jan"
    ]
    result = result.select(desired_cols)
    return result


def plot_ndvi_trends(df: pd.DataFrame):
    """
    Given a Pandas DataFrame with columns:
      - "NDVI_Onset_DOY": The day of the year when NDVI starts rising
      - "big_change_day": The day of the year with the largest NDVI percentage change
      - "year": The corresponding year

    This function:
      - Computes the mean of "NDVI_Onset_DOY" and "big_change_day" for each year.
      - Plots both trends on the same plot with the X-axis representing the years
        and the Y-axis representing the days of the year (from 200 to 350).
      - Labels the Y-axis with corresponding months.
    """

    # Compute yearly mean values for NDVI_Onset_DOY and big_change_day
    yearly_means = df.groupby("year").agg({
        "NDVI_Onset_DOY": "mean",
        "big_change_day": "mean"
    }).reset_index()

    # Define the y-axis tick labels corresponding to months
    days_range = np.arange(230, 351, 10)  # Days from 200 to 350
    month_labels = [
        "18Aug", "28Aug", "7Sep", "17Sep", "27Sep", "7ct", "17Oct", "27Oct", "6Nov", "16Nov",
        "26Nov", "6Dec", "16Dec"
    ]  # Approximate month labels for each tick

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot NDVI_Onset_DOY trend
    plt.plot(yearly_means["year"], yearly_means["NDVI_Onset_DOY"],
             marker='o', linestyle='-', color='b', label="Mean NDVI Onset DOY")

    # Plot big_change_day trend
    plt.plot(yearly_means["year"], yearly_means["big_change_day"],
             marker='s', linestyle='-', color='r', label="Mean Big Change Day")

    # Set plot labels and title
    plt.xlabel("Year")
    plt.ylabel("Day of Year")
    plt.title("Trends of NDVI Onset and Big Change Day Over the Years")
    plt.xticks(yearly_means["year"], rotation=45)
    plt.yticks(days_range, month_labels)  # Map days to months
    plt.ylim(200, 350)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    # Show the plot
    plt.show()
    

def plot_4_graphs_smooth(df: pd.DataFrame) -> None:
    """
    Splits the monthly averages into two groups:
      - Group 1: October to March (for Oct–Dec, SeasonYear = Year+1; for Jan–Mar, SeasonYear = Year)
      - Group 2: April to September (SeasonYear = Year)
    
    Computes the monthly averages of TX(homog) and TN(homog) for each SeasonYear,
    smooths the time series using spline interpolation, and plots 4 graphs:
      Graph 1: TX (Group 1: Oct-Mar)
      Graph 2: TN (Group 1: Oct-Mar)
      Graph 3: TX (Group 2: Apr-Sep)
      Graph 4: TN (Group 2: Apr-Sep)
    
    Parameters
    ----------
    df : pd.DataFrame
        Must include at least:
           - "Date": as a string ("YYYY-MM-DD") or datetime
           - "TX(homog)": numeric values
           - "TN(homog)": numeric values
    """
    # Ensure Date is datetime.
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    
    # Extract Year and Month.
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    # ---- Group 1: October to March ----
    group1 = df[df["Month"].isin([10, 11, 12, 1, 2, 3])].copy()
    # For months 10-12, assign SeasonYear = Year + 1, so that Oct-Dec 2020
    # joins Jan-Mar 2021.
    group1["SeasonYear"] = group1.apply(
        lambda row: row["Year"] + 1 if row["Month"] in [10, 11, 12] else row["Year"],
        axis=1
    )
    # Compute monthly averages.
    group1_avg = group1.groupby(["SeasonYear", "Month"])[["TX(homog)", "TN(homog)"]].mean().reset_index()
    # Pivot so rows = SeasonYear and columns = Month.
    pivot_tx_group1 = group1_avg.pivot(index="SeasonYear", columns="Month", values="TX(homog)")
    pivot_tn_group1 = group1_avg.pivot(index="SeasonYear", columns="Month", values="TN(homog)")
    # Desired order for Group1 months: October, November, December, January, February, March.
    order_group1 = [10, 11, 12, 1, 2, 3]

    # ---- Group 2: April to September ----
    group2 = df[df["Month"].isin([4, 5, 6, 7, 8, 9])].copy()
    group2["SeasonYear"] = group2["Year"]  # Use calendar year.
    group2_avg = group2.groupby(["SeasonYear", "Month"])[["TX(homog)", "TN(homog)"]].mean().reset_index()
    pivot_tx_group2 = group2_avg.pivot(index="SeasonYear", columns="Month", values="TX(homog)")
    pivot_tn_group2 = group2_avg.pivot(index="SeasonYear", columns="Month", values="TN(homog)")
    # Order for Group2 months: April, May, June, July, August, September.
    order_group2 = [4, 5, 6, 7, 8, 9]

    # ---- Define a smoothing function using spline interpolation ----
    def smooth_line(x, y, num_points=300, spline_order=3):
        """Return smooth x and y arrays after filtering non-finite values."""
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]
        if len(x_clean) < spline_order + 1:
            return x, y
        x_new = np.linspace(x_clean.min(), x_clean.max(), num_points)
        spline = make_interp_spline(x_clean, y_clean, k=spline_order)
        y_new = spline(x_new)
        return x_new, y_new

    # ---- Create a 2x2 grid for 4 graphs ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Graph 1: TX (Group 1: Oct-Mar)
    for month in order_group1:
        if month in pivot_tx_group1.columns:
            x = pivot_tx_group1.index.values.astype(float)
            y = pivot_tx_group1[month].values.astype(float)
            x_smooth, y_smooth = smooth_line(x, y)
            axes[0, 0].plot(x_smooth, y_smooth, label=calendar.month_abbr[month], linewidth=2)
    axes[0, 0].set_title("TX(homog) - October to March")
    axes[0, 0].set_xlabel("Season Year")
    axes[0, 0].set_ylabel("TX(homog)")
    axes[0, 0].legend()

    # Graph 2: TN (Group 1: Oct-Mar)
    for month in order_group1:
        if month in pivot_tn_group1.columns:
            x = pivot_tn_group1.index.values.astype(float)
            y = pivot_tn_group1[month].values.astype(float)
            x_smooth, y_smooth = smooth_line(x, y)
            axes[0, 1].plot(x_smooth, y_smooth, label=calendar.month_abbr[month], linewidth=2)
    axes[0, 1].set_title("TN(homog) - October to March")
    axes[0, 1].set_xlabel("Season Year")
    axes[0, 1].set_ylabel("TN(homog)")
    axes[0, 1].legend()

    # Graph 3: TX (Group 2: Apr-Sep)
    for month in order_group2:
        if month in pivot_tx_group2.columns:
            x = pivot_tx_group2.index.values.astype(float)
            y = pivot_tx_group2[month].values.astype(float)
            x_smooth, y_smooth = smooth_line(x, y)
            axes[1, 0].plot(x_smooth, y_smooth, label=calendar.month_abbr[month], linewidth=2)
    axes[1, 0].set_title("TX(homog) - April to September")
    axes[1, 0].set_xlabel("Season Year")
    axes[1, 0].set_ylabel("TX(homog)")
    axes[1, 0].legend()

    # Graph 4: TN (Group 2: Apr-Sep)
    for month in order_group2:
        if month in pivot_tn_group2.columns:
            x = pivot_tn_group2.index.values.astype(float)
            y = pivot_tn_group2[month].values.astype(float)
            x_smooth, y_smooth = smooth_line(x, y)
            axes[1, 1].plot(x_smooth, y_smooth, label=calendar.month_abbr[month], linewidth=2)
    axes[1, 1].set_title("TN(homog) - April to September")
    axes[1, 1].set_xlabel("Season Year")
    axes[1, 1].set_ylabel("TN(homog)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()