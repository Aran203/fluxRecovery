# SMAP + Field Data Depth-Aware, Trend-Normalized Interpolation
# With Dynamic or Fixed Weighting & Full Logging

import os
import h5py
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------

LATITUDE = 30.5
LONGITUDE = -96.5
START_DATE = "2018-01-01"
END_DATE = "2018-01-5"

USERNAME = "your_earthdata_username"
PASSWORD = "your_earthdata_password"

EXCEL_FILENAME = "RealData.xlsx"      # Field data file
TIMESTAMP_COLUMN = "Timestamp"        # Timestamp col
FIELD_COLUMN_NAME = "VWC_1_Avg"       # e.g. 5 cm
DEPTH_CM = 5

# Weighting mode: "fixed" or "dynamic"
WEIGHTING_MODE = "dynamic"
FIXED_WEIGHTS = (0.3, 0.7)  # (satellite, field)

# Dynamic weighting parameters
ROLLING_WINDOW_DAYS = 3
BETA = 1
EPSILON = 1e-6

RAW_CSV = "SMAP_SoilMoisture_Timeseries.csv"
FINAL_CSV = "Final_Reconstructed_Time_Series.csv"

# Determine which SMAP variable to use
if DEPTH_CM <= 5:
    SMAP_VARIABLE = "sm_surface"
    sat_layer_desc = "Surface (0-5 cm)"
elif DEPTH_CM >= 15:
    SMAP_VARIABLE = "sm_rootzone"
    sat_layer_desc = "Root Zone (~0-100 cm)"
else:
    SMAP_VARIABLE = "sm_surface"
    sat_layer_desc = "Surface fallback (no exact match)"
    print(f"Note: No exact SMAP match for {DEPTH_CM} cm. Using surface as fallback.")

# ------------------------------------------------------------
# NASA CMR API
# ------------------------------------------------------------
CMR_API_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

def find_smap_file(date):
    """Query NASA CMR for SMAP L4 data on the given date."""
    params = {
        "short_name": "SPL4SMGP",
        "version": "007",
        "temporal": f"{date}T00:00:00Z,{date}T23:59:59Z",
        "page_size": 1,
        "sort_key": "-start_date",
    }
    response = requests.get(CMR_API_URL, params=params)
    if response.status_code == 200:
        granules = response.json().get("feed", {}).get("entry", [])
        if granules:
            return granules[0]["links"][0]["href"]
    return None

def download_smap_data(url, filename):
    print(f"Downloading SMAP file for date: {filename.split('.')[0]}...")
    response = requests.get(url, stream=True, auth=(USERNAME, PASSWORD))
    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename}")
        return True
    else:
        print(f"Failed to download {filename} (HTTP {response.status_code})")
        return False

def extract_smap_data(filename, lat, lon, date):
    """Extract the chosen SMAP variable (surface or rootzone) at lat/lon."""
    with h5py.File(filename, "r") as f:
        latitudes = f["cell_lat"][:]
        longitudes = f["cell_lon"][:]
        soil_data = f[f"Geophysical_Data/{SMAP_VARIABLE}"][:]
        lat_idx, lon_idx = np.unravel_index(
            np.argmin(np.abs(latitudes - lat) + np.abs(longitudes - lon)),
            latitudes.shape
        )
        return {"Date": date, "Satellite_Anomaly": soil_data[lat_idx, lon_idx]}

# ------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------

print("--------------------------------------------------")
print(f"SMAP Data Fusion for Depth: {DEPTH_CM} cm")
print(f"Satellite source layer: {SMAP_VARIABLE} ({sat_layer_desc})")
print(f"Date range: {START_DATE} to {END_DATE}")
print("--------------------------------------------------")

start_dt = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE)

# 1) Download and Extract SMAP
print("Starting SMAP data download...")
all_data = []
current_dt = start_dt
file_count = 0

while current_dt <= end_dt:
    date_str = current_dt.strftime("%Y-%m-%d")
    try:
        url = find_smap_file(date_str)
        if not url:
            print(f"No SMAP file found for {date_str}")
            current_dt += timedelta(days=1)
            continue
        
        filename = url.split("/")[-1]
        if not download_smap_data(url, filename):
            current_dt += timedelta(days=1)
            continue
        
        entry = extract_smap_data(filename, LATITUDE, LONGITUDE, date_str)
        all_data.append(entry)
        os.remove(filename)
        print(f"Deleted temporary file: {filename}")
        file_count += 1
        
    except Exception as e:
        print(f"Error on {date_str}: {e}")
    
    current_dt += timedelta(days=1)

sat_df = pd.DataFrame(all_data)
sat_df["Date"] = pd.to_datetime(sat_df["Date"])

print("--------------------------------------------------")
print(f"Downloaded SMAP files for {file_count} days within {START_DATE} - {END_DATE}.")
sat_df.to_csv(RAW_CSV, index=False)
print(f"Satellite data (trend only) saved to {RAW_CSV}")

# 2) Load Field Data (Raw + Daily)
print("Loading field data from Excel...")
raw_field_df = pd.read_excel(EXCEL_FILENAME)
raw_field_df = raw_field_df.rename(columns={TIMESTAMP_COLUMN: "Date", FIELD_COLUMN_NAME: "Field_Measurement"})
raw_field_df["Date"] = pd.to_datetime(raw_field_df["Date"])

# Keep only relevant date range in raw
raw_field_df = raw_field_df[(raw_field_df["Date"] >= start_dt) & (raw_field_df["Date"] <= end_dt)]

# Make daily-averaged field data
daily_field_df = raw_field_df.set_index("Date").resample("D").mean(numeric_only=True).reset_index()

# 3) Normalize Satellite Trend
field_avg = (daily_field_df["Field_Measurement"].max() + daily_field_df["Field_Measurement"].min()) / 2
smap_avg = (sat_df["Satellite_Anomaly"].max() + sat_df["Satellite_Anomaly"].min()) / 2
sat_df["Satellite_Anomaly"] += (field_avg - smap_avg)

# 4) Merge and Determine Weights
combined_df = pd.merge(sat_df, daily_field_df, on="Date", how="outer").sort_values("Date")

# Rolling std for dynamic
sat_rolling_std = combined_df["Satellite_Anomaly"].rolling(window=ROLLING_WINDOW_DAYS, min_periods=1, center=True).std()
ground_rolling_std = combined_df["Field_Measurement"].rolling(window=ROLLING_WINDOW_DAYS, min_periods=1, center=True).std()

if WEIGHTING_MODE == "dynamic":
    ratio = sat_rolling_std / (ground_rolling_std + EPSILON)
    combined_df["w_sat"] = 1 / (1 + (ratio ** BETA))
    combined_df["w_ground"] = 1 - combined_df["w_sat"]
else:
    w_sat, w_ground = FIXED_WEIGHTS
    combined_df["w_sat"] = w_sat
    combined_df["w_ground"] = w_ground

# 5) Weighted Interpolation
def interpolate_weighted(df):
    df = df.copy().reset_index(drop=True)
    df["Interpolated_Value"] = df["Field_Measurement"]  # Start
    for i in range(len(df)):
        if pd.isna(df.loc[i, "Field_Measurement"]):
            sat_val = df.loc[i, "Satellite_Anomaly"]
            # Attempt to get a linear field interpolation
            interp = df["Field_Measurement"].interpolate("linear").loc[i]
            wsat = df.loc[i, "w_sat"]
            wgrd = df.loc[i, "w_ground"]
            if not pd.isna(sat_val) and not pd.isna(interp):
                df.loc[i, "Interpolated_Value"] = wsat * sat_val + wgrd * interp
            else:
                df.loc[i, "Interpolated_Value"] = interp
    df["Interpolated_Value"] = df["Interpolated_Value"].interpolate("linear", limit_direction="both")
    return df

combined_df = interpolate_weighted(combined_df)

# 6) Make Final Daily DF
full_dates = pd.date_range(start=start_dt, end=end_dt, freq="D")
df_final = pd.DataFrame({"Date": full_dates})
df_final = df_final.merge(combined_df[["Date", "Interpolated_Value"]], on="Date", how="left")
df_final["Interpolated_Value"] = df_final["Interpolated_Value"].interpolate("linear", limit_direction="both")

# 7) Stats on Weighted Approach
if WEIGHTING_MODE == "dynamic":
    mean_ws = combined_df["w_sat"].mean()
    min_ws = combined_df["w_sat"].min()
    max_ws = combined_df["w_sat"].max()
    print("--------------------------------------------------")
    print("Dynamic Weighting Stats:")
    print(f"Rolling window: {ROLLING_WINDOW_DAYS} days | Beta={BETA} | Epsilon={EPSILON}")
    print(f"Mean satellite weight: {mean_ws:.3f} | Min: {min_ws:.3f} | Max: {max_ws:.3f}")
    print(f"Mean ground weight: {1-mean_ws:.3f}")
elif WEIGHTING_MODE == "fixed":
    print("--------------------------------------------------")
    print(f"Fixed weights: Satellite={FIXED_WEIGHTS[0]}, Ground={FIXED_WEIGHTS[1]}")

# 8) Save Final CSV
df_final.to_csv(FINAL_CSV, index=False)
print("--------------------------------------------------")
print(f"Final daily time series saved to {FINAL_CSV}")

# Field data coverage
valid_days = daily_field_df["Field_Measurement"].notna().sum()
total_days = len(daily_field_df)
print(f"Daily field data coverage: {valid_days} / {total_days} days have measurements.")

# 9) Plot
print("Plotting result now...")
plt.figure(figsize=(12, 6))

# Green line: Interpolated time series
plt.plot(df_final["Date"], df_final["Interpolated_Value"], color='green', label="Interpolated Time Series")

# Blue dots: daily-averaged field data
daily_mask = combined_df["Field_Measurement"].notna()
plt.scatter(
    combined_df.loc[daily_mask, "Date"], 
    combined_df.loc[daily_mask, "Field_Measurement"], 
    color='blue', 
    label="Daily Field Data"
)

# Orange dots: satellite trend
sat_mask = combined_df["Satellite_Anomaly"].notna()
plt.scatter(
    combined_df.loc[sat_mask, "Date"], 
    combined_df.loc[sat_mask, "Satellite_Anomaly"], 
    color='orange', 
    label="Satellite Trend"
)

# Gray dots: raw 30-min data
raw_mask = raw_field_df["Field_Measurement"].notna()
plt.scatter(
    raw_field_df.loc[raw_mask, "Date"], 
    raw_field_df.loc[raw_mask, "Field_Measurement"], 
    color='gray', alpha=0.5, s=10, 
    label="Raw 30-min Field Data"
)

plt.xlabel("Date")
plt.ylabel("Soil Moisture (m³/m³)")
plt.title(f"Interpolated Soil Moisture at {DEPTH_CM} cm - {WEIGHTING_MODE.capitalize()} Weighting")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("--------------------------------------------------")
print(f"Process complete for {START_DATE} to {END_DATE} at depth {DEPTH_CM} cm.")
print("Plot displayed with daily field data, raw 30-min field data, satellite trend, and final interpolation.")
print("--------------------------------------------------")
