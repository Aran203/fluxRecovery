import os
import h5py
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ----------------------------------------
# USER CONFIGURATION
# ----------------------------------------
LATITUDE = 30.5
LONGITUDE = -96.5
START_DATE = "2018-01-01"
END_DATE = "2018-01-31"
USERNAME = "your_earthdata_username"
PASSWORD = "your_earthdata_password"

RAW_CSV = "SMAP_SoilMoisture_Timeseries.csv"
FINAL_CSV = "Final_Reconstructed_Time_Series.csv"

# ----------------------------------------
# CMR API Endpoint for SMAP
# ----------------------------------------
CMR_API_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

# ----------------------------------------
# Find SMAP File from NASA CMR
# ----------------------------------------
def find_smap_file(date):
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

# ----------------------------------------
# Download File from URL
# ----------------------------------------
def download_smap_data(url, filename):
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✔ Downloaded: {filename}")
        return True
    else:
        print(f"✘ Download failed with code {response.status_code}")
        return False

# ----------------------------------------
# Extract Soil Moisture from HDF5 File
# ----------------------------------------
def extract_smap_data(filename, lat, lon, date):
    with h5py.File(filename, "r") as f:
        latitudes = f["cell_lat"][:]
        longitudes = f["cell_lon"][:]
        sm_surface = f["Geophysical_Data/sm_surface"][:]

        lat_idx, lon_idx = np.unravel_index(
            np.argmin(np.abs(latitudes - lat) + np.abs(longitudes - lon)),
            latitudes.shape
        )
        value = sm_surface[lat_idx, lon_idx]
        return {"Date": date, "Latitude": lat, "Longitude": lon,
                "Surface Moisture (m³/m³)": value}

# ----------------------------------------
# Download and Extract for Each Day
# ----------------------------------------
all_data = []
start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
current_dt = start_dt

while current_dt <= end_dt:
    date_str = current_dt.strftime("%Y-%m-%d")
    try:
        url = find_smap_file(date_str)
        if not url:
            print(f"No file for {date_str}")
            current_dt += timedelta(days=1)
            continue

        filename = url.split("/")[-1]
        if not download_smap_data(url, filename):
            current_dt += timedelta(days=1)
            continue

        entry = extract_smap_data(filename, LATITUDE, LONGITUDE, date_str)
        all_data.append(entry)
        os.remove(filename)
        print(f"🗑️ Deleted file: {filename}")

    except Exception as e:
        print(f"⚠️ Error on {date_str}: {e}")

    current_dt += timedelta(days=1)

# ----------------------------------------
# Save Raw CSV
# ----------------------------------------
df_raw = pd.DataFrame(all_data)
df_raw.to_csv(RAW_CSV, index=False)
print(f"✔ Raw data saved to {RAW_CSV}")

# ----------------------------------------
# Interpolate with Field-Priority Logic
# ----------------------------------------
df = df_raw.copy()
df["Date"] = pd.to_datetime(df["Date"])
df = df.rename(columns={"Surface Moisture (m³/m³)": "Field_Measurement"})
df["Satellite_Anomaly"] = df["Field_Measurement"]

def process_data_with_field_priority(df):
    df = df.sort_values("Date")
    df["Value"] = df["Field_Measurement"].combine_first(df["Satellite_Anomaly"])

    df["Interpolated_Value"] = df["Field_Measurement"]
    for i in range(len(df)):
        if pd.isna(df.loc[i, "Field_Measurement"]):
            sat_val = df.loc[i, "Satellite_Anomaly"]
            interp_field = df["Field_Measurement"].interpolate("linear").loc[i]
            df.loc[i, "Interpolated_Value"] = 0.3 * sat_val + 0.7 * interp_field

    df["Interpolated_Value"] = df["Interpolated_Value"].interpolate("linear", limit_direction="both")
    return df

df_interp = process_data_with_field_priority(df)

# ----------------------------------------
# Save Final Clean Time Series
# ----------------------------------------
full_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
df_final = pd.DataFrame({"Date": full_dates})
df_final = df_final.merge(df_interp[["Date", "Interpolated_Value"]], on="Date", how="left")
df_final["Interpolated_Value"] = df_final["Interpolated_Value"].interpolate("linear", limit_direction="both")
df_final.to_csv(FINAL_CSV, index=False)
print(f"✔ Final interpolated series saved to {FINAL_CSV}")

# ----------------------------------------
# Plot Final Time Series
# ----------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(df_final["Date"], df_final["Interpolated_Value"], color='green', linewidth=2, label="Interpolated SMAP Series")
plt.xlabel("Date")
plt.ylabel("Soil Moisture (m³/m³)")
plt.title("Interpolated SMAP Soil Moisture Time Series")
plt.legend()
plt.grid(True)
plt.show()
