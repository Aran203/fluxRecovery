import os
import h5py
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ----------------------------------------
# USER CONFIGURATION
# ----------------------------------------
LATITUDE = 30.5
LONGITUDE = -96.5
START_DATE = "2018-01-01"
END_DATE = "2018-01-31"
USERNAME = "your_earthdata_username"
PASSWORD = "your_earthdata_password"
node_number = 1  # 1, 2, or 3
excel_file = "RealData.xlsx"

# Output CSV name
CSV_FILENAME = "Final_Reconstructed_Time_Series.csv"

# ----------------------------------------
# NASA CMR API Endpoint for SMAP L4
# ----------------------------------------
CMR_API_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

# ----------------------------------------
# FUNCTION: Find SMAP File for a Date
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
# FUNCTION: Download SMAP File
# ----------------------------------------
def download_smap_data(url, filename):
    print(f"Downloading SMAP data from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✔ Downloaded: {filename}")
        return True
    else:
        print(f"✘ Failed download: {response.status_code}")
        return False

# ----------------------------------------
# FUNCTION: Extract Soil Moisture
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
        return {"Date": date, "Satellite_Anomaly": value}

# ----------------------------------------
# FUNCTION: Load Real Field Data
# ----------------------------------------
def load_field_data(excel_file, node_number):
    df = pd.read_excel(excel_file)
    df["Date"] = pd.to_datetime(df["Date"])
    if node_number < 1 or node_number > 3:
        raise ValueError("Node number must be 1, 2, or 3.")
    column_name = df.columns[node_number]
    return df[["Date", column_name]].rename(columns={column_name: "Field_Measurement"})

# ----------------------------------------
# FUNCTION: Field Priority Interpolation
# ----------------------------------------
def process_data_with_field_priority(satellite_df, field_df):
    df = pd.merge(satellite_df, field_df, on="Date", how="outer").sort_values("Date")
    df["Value"] = df["Field_Measurement"].combine_first(df["Satellite_Anomaly"])

    df["Interpolated_Value"] = df["Field_Measurement"]
    for i in range(len(df)):
        if pd.isna(df.loc[i, "Field_Measurement"]):
            sat_val = df.loc[i, "Satellite_Anomaly"]
            interp_field = df["Field_Measurement"].interpolate("linear").loc[i]
            df.loc[i, "Interpolated_Value"] = 0.3 * sat_val + 0.7 * interp_field

    df["Interpolated_Value"] = df["Interpolated_Value"].interpolate("linear", limit_direction="both")
    return df

# ----------------------------------------
# FUNCTION: Save to CSV
# ----------------------------------------
def save_final_series(df, filename):
    full_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    full_df = pd.DataFrame({"Date": full_dates})
    full_df = full_df.merge(df[["Date", "Interpolated_Value"]], on="Date", how="left")
    full_df["Interpolated_Value"] = full_df["Interpolated_Value"].interpolate("linear", limit_direction="both")
    full_df.to_csv(filename, index=False)
    print(f"✔ Final time series saved to: {filename}")

# ----------------------------------------
# MAIN EXECUTION
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
            print(f"No data for {date_str}")
            current_dt += timedelta(days=1)
            continue

        filename = url.split("/")[-1]
        if not download_smap_data(url, filename):
            current_dt += timedelta(days=1)
            continue

        entry = extract_smap_data(filename, LATITUDE, LONGITUDE, date_str)
        all_data.append(entry)
        os.remove(filename)

    except Exception as e:
        print(f"⚠️ Error on {date_str}: {e}")

    current_dt += timedelta(days=1)

# Convert to DataFrame
satellite_df = pd.DataFrame(all_data)
satellite_df["Date"] = pd.to_datetime(satellite_df["Date"])

# Load field data from Excel
field_df = load_field_data(excel_file, node_number)

# Process & Interpolate
combined_df = process_data_with_field_priority(satellite_df, field_df)

# Save final time series
save_final_series(combined_df, CSV_FILENAME)

# Plot result
plt.figure(figsize=(12, 6))
plt.plot(combined_df["Date"], combined_df["Interpolated_Value"], color='green', linewidth=2, label="SMAP Reconstructed Series")
plt.xlabel("Date")
plt.ylabel("Interpolated Soil Moisture (m³/m³)")
plt.title("SMAP + Field Blended Soil Moisture")
plt.legend()
plt.grid(True)
plt.show()
