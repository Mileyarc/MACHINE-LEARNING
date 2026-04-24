import os
import requests
import json
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.getcwd())
DATA_RAW = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT_DIR, "data", "processed")

# Ensure directories exist
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)

CITY = "Hanoi"

WEATHER_URL = "https://api.weatherbit.io/v2.0/history/hourly"
AQI_URL = "https://api.weatherbit.io/v2.0/history/airquality"

START_DATE = "2022-01-13"
END_DATE = "2026-03-13"

# Key is pulled from environment variable for security
# Multiple keys can be set in the format WEATHER_API_KEY_1, WEATHER_API_KEY_2, etc.
API_KEY = os.getenv("WEATHER_API_KEY_1", os.getenv("WEATHER_API_KEY_2"))  

FILES = {
    "weather_raw": os.path.join(DATA_RAW, "weather_data_final.json"),
    "aqi_raw": os.path.join(DATA_RAW, "aqi_data_final.json"),
    "data_merged": os.path.join(DATA_PROCESSED, "data_merged.csv"),
    "data_cleaned": os.path.join(DATA_PROCESSED, "data_cleaned.csv"),
}

def fetch_data(url_base, output_file, data_label="Data"):
    """
    Consolidated crawler logic using a unified date-range approach.
    """
    data_total = {}
    try:
        with open(output_file, 'r') as f:
            data_total = json.load(f)
            print(f"Resuming {data_label} from {len(data_total.get('data', []))} records in {output_file}")
    except FileNotFoundError:
        print(f"No existing {data_label} file found, starting fresh.")
    
    current_start = pd.to_datetime(START_DATE)
    final_end = pd.to_datetime(END_DATE)

    while current_start < final_end:
        current_end = current_start + pd.DateOffset(months=1)
        if current_end > final_end:
            current_end = final_end
        
        # Skip if data for this period might already exist
        if data_total and 'data' in data_total and any(record['timestamp_utc'] > current_start.isoformat() and record['timestamp_utc'] < current_end.isoformat() for record in data_total['data']):
            print(f"Data for {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')} likely already exists, skipping.")
            current_start = current_end
            continue

        params = {
            "city": CITY,
            "key": API_KEY,
            "tz": "local",
            "start_date": current_start.strftime('%Y-%m-%d'),
            "end_date": current_end.strftime('%Y-%m-%d')
        }

        try:
            res = requests.get(url_base, params=params)
            res.raise_for_status()

            if not data_total:
                # Capture the full first response
                data_total = res.json()
                print(f"Initialized total {data_label} with metadata for {CITY}")
            else:
                if 'data' in res.json():
                    data_total['data'].extend(res.json().get('data', []))
            
            print(f"Fetched {data_label} for {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}, total records: {len(data_total['data'])}")
        except Exception as e:
            print(f"Error fetching {data_label} for {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}: {e}")
        
        current_start = current_end
    
    with open(output_file, 'w') as f:
        json.dump(data_total, f, indent=4)

def deduplicate_data(json_file):
    '''
    Ensures data integrity by removing duplicate records.
    - If duplicates have identical data, retain one and discard the rest.
    - If duplicates have differing data, retain similar fields and mark differing fields as 'N/A'.
    '''
    if not os.path.exists(json_file):
        print(f"File {json_file} not found. Skipping deduplication.")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    data_list = data.get('data', [])
    if not data_list:
        print(f"No data found in {json_file}. Skipping deduplication.")
        return
    
    df = pd.json_normalize(data_list)
    
    # Find all rows that share a timestamp
    all_dupes = df[df.duplicated(subset=['timestamp_utc'], keep=False)]

    if not all_dupes.empty:
        # Print the number of duplicate timestamps found
        num_dupes = all_dupes['timestamp_utc'].nunique()
        print(f"Found {num_dupes} duplicate timestamps in {json_file}. Proceeding with deduplication.")

        # Check if any column in the group has more than 1 unique value
        conflicts = all_dupes.groupby('timestamp_utc').filter(lambda x: (x.nunique() > 1).any())
        if not conflicts.empty:
            num_conflicts = conflicts['timestamp_utc'].nunique()
            print(f"Found {num_conflicts} conflicting timestamps in {json_file}.")

            with open(f'{json_file.split(".")[0]}_conflicts.json', 'w') as f:
                json.dump(conflicts.to_dict(orient='records'), f, indent=4)
    else:
        print(f"No duplicates found in {json_file}. No deduplication needed.")
        print(f"Total records in {json_file}: {len(df)}")
        return df
    
    # Count unique values for every column per timestamp
    counts = df.groupby('timestamp_utc').transform('nunique')

    # Identify the first occurence of each timestamp to keep as base
    df_resolved = df.drop_duplicates(subset=['timestamp_utc'], keep='first').copy()

    # Re-index the counts to match the df_resolved
    counts_resolved = counts.loc[df_resolved.index]

    # Mask the values: If unique count > 1, set to None
    for col in df_resolved.columns:
        if col != 'timestamp_utc':
            df_resolved.loc[counts_resolved[col] > 1, col] = None

    data['data'] = df_resolved.replace({np.nan: None}).to_dict(orient='records')
    with open(f'{json_file.split(".")[0]}_deduplicated.json', 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Deduplicated data saved to {json_file.split('.')[0]}_deduplicated.json with {len(df_resolved)} unique records.")

    return df_resolved

def merge_dataframes(df_weather, df_aqi, output_file=FILES['data_merged']):
    '''
    Merges weather and AQI DataFrames on timestamp, ensuring no duplicated columns and preserving all relevant features.
    - If columns are duplicated, keep one.
    - Sort the merged DataFrame by timestamp and ensure timestamp is the first column.
    '''
    # Merge the two DataFrames on timestamp but remove duplicated columns
    df_merged = pd.merge(df_weather, df_aqi, on='timestamp_utc', suffixes=('_w', '_a'))
    # Find the columns name with the same name but different suffixes, remove one and rename the other to the original name
    for col in df_merged.columns:
        if col.endswith('_w'):
            base_col = col[:-2]
            if base_col + '_a' in df_merged.columns:
                df_merged.drop(col, axis=1, inplace=True)
                df_merged.rename(columns={base_col + '_a': base_col}, inplace=True)

    # Sort by time
    df_merged = df_merged.sort_values(by='timestamp_utc')

    # Reorder: Put local time and UTC time at the very front
    cols = df_merged.columns.tolist()
    for time_col in ['timestamp_utc', 'timestamp_local']:
        if time_col in cols:
            cols.insert(0, cols.pop(cols.index(time_col)))

    df_merged = df_merged[cols]

    # Export the merged DataFrame to a CSV file
    df_merged.to_csv(output_file, index=False)
    print(f"Merged data successfully saved to {output_file}")

    return df_merged

def preliminary_clean_data(df_merged, output_file=FILES['data_cleaned']):
    '''
    Pruning constants and metadata.
    Removes columns that provide no variance or are deprecated before proceeding to imputation and diagnostic correlation analysis.
    '''
    try:
        # Drop deprecated & UI metadata
        # Labels like 'datetime' and 'h_angle' are deprecated 
        # Columns like 'revision_status' and 'revision_version' are metadata that do not contribute to the model's learning and can be safely removed.
        # UI assets (icons/descriptions) are redundant with numerical features and can be removed to reduce noise.
        metadata_cols = [col for col in ['datetime', 'h_angle', 'revision_status', 'revision_version', 'weather.code', 'weather.description', 'weather.icon'] if col in df_merged.columns]
        print(f"Identified metadata columns to drop: {metadata_cols}")
        df_merged.drop(columns=metadata_cols, inplace=True)

        # Drop zero-variance columns (constants)
        # Columns like 'snow' are expected to have zero variance in Hanoi's climate, offering no predictive value.
        constant_cols = [col for col in df_merged.columns if df_merged[col].nunique() <= 1]
        print(f"Identified constant columns to drop: {constant_cols}")
        df_merged.drop(columns=constant_cols, inplace=True)

        # Drop redundant temporal formats
        # Use 'timestamp_local' as the primary time reference and drop 'timestamp_utc' and 'ts' to avoid redundancy.
        redundant_time_cols = [col for col in ['timestamp_utc', 'ts'] if col in df_merged.columns]
        print(f"Identified redundant time columns to drop: {redundant_time_cols}")
        df_merged.drop(columns=redundant_time_cols, inplace=True)

        # Save for the inputation step
        df_merged.to_csv(output_file, index=False)
        print(f"Preliminary cleaned data saved to {output_file} with {df_merged.shape[0]} records and {df_merged.shape[1]} features.")
    except Exception as e:
        print(f"An error occurred during preliminary cleaning: {e}")

if __name__ == "__main__":
    if not API_KEY:
        raise ValueError("API key not found. Please set the WEATHER_API_KEY_1 or WEATHER_API_KEY_2 environment variable.")
    else:
        # Fetch historical weather and AQI data 
        print("Starting data fetching process...")
        fetch_data(WEATHER_URL, FILES['weather_raw'], data_label="Weather Data")
        fetch_data(AQI_URL, FILES['aqi_raw'], data_label="AQI Data")
        
        # Deduplicate the raw data
        print("\nStarting deduplication process...")
        weather_deduplicate = deduplicate_data(FILES['weather_raw'])
        aqi_deduplicate = deduplicate_data(FILES['aqi_raw'])
        
        # Merge and preliminary clean the data for analysis
        print("\nStarting merging and preliminary cleaning process...")
        df_merged = merge_dataframes(weather_deduplicate, aqi_deduplicate)
        preliminary_clean_data(df_merged)