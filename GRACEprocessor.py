import subprocess
import sys
import os

# ----------------------------------------
# Install Missing Packages
# ----------------------------------------
def install_packages():
    required_packages = ["requests", "pandas", "numpy", "matplotlib", "netCDF4", "openpyxl"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages()

# ----------------------------------------
# Import Libraries
# ----------------------------------------
import netCDF4 as nc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime

# ----------------------------------------
# Download NetCDF File
# ----------------------------------------
def download_nc_file(username, password, url, output_file):
    print("Downloading NetCDF file...")
    try:
        with requests.get(url, auth=(username, password), stream=True) as response:
            response.raise_for_status()
            with open(output_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        print(f"File downloaded: {output_file}")
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")

# ----------------------------------------
# Load Satellite Data
# ----------------------------------------
def load_satellite_data(nc_file, start_year, end_year, latitude, longitude):
    print("Loading satellite data...")
    dataset = nc.Dataset(nc_file, 'r')

    latitudes = dataset.variables['lat'][:]
    longitudes = dataset.variables['lon'][:]
    time_var = dataset.variables['time'][:]
    lwe_thickness = dataset.variables['lwe_thickness'][:]

    time_units = dataset.variables['time'].units
    dates = nc.num2date(time_var, units=time_units)
    dates = [datetime(d.year, d.month, d.day) for d in dates]

    lat_idx = np.argmin(np.abs(latitudes - latitude))
    lon_idx = np.argmin(np.abs(longitudes - longitude))

    satellite_data = []
    time_labels = []
    for i, date in enumerate(dates):
        if start_year <= date.year <= end_year:
            satellite_data.append(lwe_thickness[i, lat_idx, lon_idx])
            time_labels.append(date)

    dataset.close()

    satellite_df = pd.DataFrame({
        "Date": time_labels,
        "Satellite_Anomaly": satellite_data
    })

    output_file = f"Satellite_Data_{start_year}_{end_year}.csv"
    satellite_df.to_csv(output_file, index=False)
    print(f"Satellite data saved to {output_file}")
    return satellite_df

# ----------------------------------------
# Load Field Data
# ----------------------------------------
def load_field_data(excel_file, node_number):
    print("Loading field data...")
    field_data = pd.read_excel(excel_file)
    field_data["Date"] = pd.to_datetime(field_data["Date"])

    if node_number < 1 or node_number > 3:
        raise ValueError("Node number must be 1, 2, or 3.")

    node_column = field_data.columns[node_number]
    field_df = field_data[["Date", node_column]].rename(columns={node_column: "Field_Measurement"})

    print(f"Field data for Node {node_number} loaded.")
    return field_df

# ----------------------------------------
# Normalize Satellite Data to Field Range
# ----------------------------------------
def normalize_satellite_to_field(satellite_df, field_df):
    print("Normalizing satellite data...")
    field_avg = (field_df["Field_Measurement"].max() + field_df["Field_Measurement"].min()) / 2
    sat_avg = (satellite_df["Satellite_Anomaly"].max() + satellite_df["Satellite_Anomaly"].min()) / 2
    adjustment = field_avg - sat_avg
    satellite_df["Satellite_Anomaly"] += adjustment

    satellite_df.to_csv("Adjusted_Satellite_Data.csv", index=False)
    return satellite_df

# ----------------------------------------
# Merge & Interpolate Data with Field Priority
# ----------------------------------------
def process_data_with_field_priority(satellite_df, field_df):
    print("Combining and interpolating data...")
    combined_df = pd.merge(satellite_df, field_df, on="Date", how="outer").sort_values("Date")
    combined_df['Value'] = combined_df['Field_Measurement'].combine_first(combined_df['Satellite_Anomaly'])

    # Interpolate missing values using a weighted blend
    combined_df['Interpolated_Value'] = combined_df['Field_Measurement']

    for i in range(len(combined_df)):
        if pd.isna(combined_df.loc[i, 'Field_Measurement']):
            sat_val = combined_df.loc[i, 'Satellite_Anomaly']
            interpolated_field = combined_df['Field_Measurement'].interpolate(method='linear').loc[i]
            combined_df.loc[i, 'Interpolated_Value'] = 0.3 * sat_val + 0.7 * interpolated_field

    # Final linear interpolation to ensure no missing values at start/end
    combined_df['Interpolated_Value'] = combined_df['Interpolated_Value'].interpolate(method='linear', limit_direction='both')

    return combined_df

# ----------------------------------------
# Save Final Reconstructed Time Series
# ----------------------------------------
def save_reconstructed_time_series(combined_df):
    print("Saving final time series...")
    date_range = pd.date_range(start=combined_df["Date"].min(), end=combined_df["Date"].max(), freq='D')
    full_series = pd.DataFrame({"Date": date_range})
    full_series = full_series.merge(combined_df[['Date', 'Interpolated_Value']], on="Date", how="left")
    
    # Interpolate across entire range (including before first field value)
    full_series["Interpolated_Value"] = full_series["Interpolated_Value"].interpolate(method='linear', limit_direction='both')

    output_file = "Final_Reconstructed_Time_Series.csv"
    full_series.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

# ----------------------------------------
# Clean Up All Except Final CSV & Excel
# ----------------------------------------
def cleanup_temp_files(keep_file="Final_Reconstructed_Time_Series.csv", keep_excel="RealData.xlsx"):
    print("Cleaning up temporary files...")
    for f in os.listdir():
        if (f.endswith(".nc") or f.endswith(".csv")) and f not in [keep_file, keep_excel]:
            try:
                os.remove(f)
                print(f"Deleted: {f}")
            except Exception as e:
                print(f"Could not delete {f}: {e}")

# ----------------------------------------
# Inputs
# ----------------------------------------
username = "your_username"
password = "your_password"
nc_url = "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4/GRCTellus.JPL.200204_202411.GLO.RL06.3M.MSCNv04CRI.nc"
nc_file = "GRCTellus_data.nc"

excel_file = "RealData.xlsx"
start_year = 2019
end_year = 2020
latitude = 28 + 46 / 60 + 49.8 / 3600
longitude = -(95 + 36 / 60 + 51.7 / 3600)
node_number = 1

# ----------------------------------------
# Run the Script
# ----------------------------------------
download_nc_file(username, password, nc_url, nc_file)
satellite_df = load_satellite_data(nc_file, start_year, end_year, latitude, longitude)
field_df = load_field_data(excel_file, node_number)
normalized_satellite_df = normalize_satellite_to_field(satellite_df, field_df)
combined_df = process_data_with_field_priority(normalized_satellite_df, field_df)
save_reconstructed_time_series(combined_df)
cleanup_temp_files()

# ----------------------------------------
# Plot the Final Time Series
# ----------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(combined_df["Date"], combined_df["Interpolated_Value"], label="Reconstructed Time Series", color='green', linewidth=2)
plt.xlabel("Date")
plt.ylabel("Interpolated Value")
plt.title("Reconstructed Time Series")
plt.legend()
plt.grid(True)
plt.show()
