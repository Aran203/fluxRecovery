import os
import requests
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ----------------------------------------
# USER CONFIGURATION
# ----------------------------------------
LATITUDE = 30.5
LONGITUDE = -96.5
START_YEAR = 2023
END_YEAR = 2023
USERNAME = "your_earthdata_username"
PASSWORD = "your_earthdata_password"
node_number = 1  # Set to 1, 2, or 3
excel_file = "RealData.xlsx"

# ----------------------------------------
# FUNCTION: Generate IMERG URL
# ----------------------------------------
def generate_imerg_url(date):
    year, month, day = date.strftime("%Y"), date.strftime("%m"), date.strftime("%d")
    filename = f"3B-DAY.MS.MRG.3IMERG.{year}{month}{day}-S000000-E235959.V07B.nc4"
    url = f"https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/{year}/{month}/{filename}"
    return url, filename

# ----------------------------------------
# FUNCTION: Download File
# ----------------------------------------
def download_imerg_data(url, local_filename, username, password):
    with requests.get(url, auth=(username, password), stream=True) as response:
        if response.status_code == 200:
            with open(local_filename, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            return True
        else:
            return False

# ----------------------------------------
# FUNCTION: Extract Precip Value
# ----------------------------------------
def extract_precipitation(filename, lat, lon, date):
    with h5py.File(filename, "r") as f:
        latitudes = f["lat"][:]
        longitudes = f["lon"][:]
        precipitation = f["precipitation"][:]  # Shape (1, lat, lon)

        lat_idx = np.argmin(np.abs(latitudes - lat))
        lon_idx = np.argmin(np.abs(longitudes - lon))
        value = precipitation[0, lat_idx, lon_idx]

        return {"Date": date, "Satellite_Anomaly": value}

# ----------------------------------------
# FUNCTION: Load Field Data
# ----------------------------------------
def load_field_data(excel_file, node_number):
    field_data = pd.read_excel(excel_file)
    field_data["Date"] = pd.to_datetime(field_data["Date"])
    if node_number < 1 or node_number > 3:
        raise ValueError("Node number must be 1, 2, or 3.")
    column_name = field_data.columns[node_number]
    return field_data[["Date", column_name]].rename(columns={column_name: "Field_Measurement"})

# ----------------------------------------
# FUNCTION: Process Field Priority
# ----------------------------------------
def process_data_with_field_priority(satellite_df, field_df):
    df = pd.merge(satellite_df, field_df, on="Date", how="outer").sort_values("Date")
    df["Value"] = df["Field_Measurement"].combine_first(df["Satellite_Anomaly"])

    # Weighted interpolation blend
    df["Interpolated_Value"] = df["Field_Measurement"]
    for i in range(len(df)):
        if pd.isna(df.loc[i, "Field_Measurement"]):
            sat_val = df.loc[i, "Satellite_Anomaly"]
            interp_field = df["Field_Measurement"].interpolate("linear").loc[i]
            df.loc[i, "Interpolated_Value"] = 0.3 * sat_val + 0.7 * interp_field

    # Final interpolation (forward + backward fill)
    df["Interpolated_Value"] = df["Interpolated_Value"].interpolate("linear", limit_direction="both")
    return df

# ----------------------------------------
# FUNCTION: Save Final CSV
# ----------------------------------------
def save_final_series(df, filename="Final_Reconstructed_Time_Series.csv"):
    full_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    full_df = pd.DataFrame({"Date": full_dates})
    full_df = full_df.merge(df[["Date", "Interpolated_Value"]], on="Date", how="left")
    full_df["Interpolated_Value"] = full_df["Interpolated_Value"].interpolate("linear", limit_direction="both")
    full_df.to_csv(filename, index=False)
    print(f"✔ Final data saved to {filename}")

# ----------------------------------------
# MAIN EXECUTION
# ----------------------------------------
all_data = []
for year in range(START_YEAR, END_YEAR + 1):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    current = start_date

    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        try:
            url, filename = generate_imerg_url(current)
            if download_imerg_data(url, filename, USERNAME, PASSWORD):
                data = extract_precipitation(filename, LATITUDE, LONGITUDE, date_str)
                all_data.append(data)
                os.remove(filename)
            else:
                print(f"✘ Failed to download {filename}")
        except Exception as e:
            print(f"⚠ Error on {date_str}: {e}")
        current += timedelta(days=1)

# Create satellite DataFrame
satellite_df = pd.DataFrame(all_data)
satellite_df["Date"] = pd.to_datetime(satellite_df["Date"])

# Load field data from Excel
field_df = load_field_data(excel_file, node_number)

# Combine and interpolate
combined_df = process_data_with_field_priority(satellite_df, field_df)

# Save final result
save_final_series(combined_df)

# Plot result
plt.figure(figsize=(12, 6))
plt.plot(combined_df["Date"], combined_df["Interpolated_Value"], color='green', linewidth=2, label="IMERG Reconstructed Series")
plt.xlabel("Date")
plt.ylabel("Interpolated Precipitation (mm/day)")
plt.title("IMERG + Field Blended Precipitation")
plt.legend()
plt.grid(True)
plt.show()
