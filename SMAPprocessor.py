import os
import h5py
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -------------------- USER SETTINGS --------------------
LATITUDE = 30.5
LONGITUDE = -96.5
START_DATE = "2018-01-01"
END_DATE = "2018-12-31"  
USERNAME = "your_earthdata_username"
PASSWORD = "your_earthdata_password"

EXCEL_FILENAME = "RealData.xlsx"      # Field data file from Excel
TIMESTAMP_COLUMN = "Timestamp"        # Name of timestamp column in Excel
FIELD_COLUMN_NAME = "VWC_1_Avg"         # Field measurement (e.g., 5 cm)
DEPTH_CM = 5

FINAL_CSV = "Final_Reconstructed_Time_Series.csv"  # Final CSV will contain only Date and Final

# -------------------- SMAP SETUP --------------------
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

CMR_API_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

# -------------------- NASA CMR API FUNCTIONS --------------------
def find_smap_file(date_str):
    params = {
        "short_name": "SPL4SMGP",
        "version": "007",
        "temporal": f"{date_str}T00:00:00Z,{date_str}T23:59:59Z",
        "page_size": 1,
        "sort_key": "-start_date",
    }
    r = requests.get(CMR_API_URL, params=params)
    if r.status_code == 200:
        entries = r.json().get("feed", {}).get("entry", [])
        if entries:
            return entries[0]["links"][0]["href"]
    return None

def download_smap_file(url, filename):
    r = requests.get(url, stream=True, auth=(USERNAME, PASSWORD))
    if r.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False

def extract_smap_value(fname, lat, lon, date_str):
    with h5py.File(fname, "r") as f:
        lats = f["cell_lat"][:]
        lons = f["cell_lon"][:]
        data = f[f"Geophysical_Data/{SMAP_VARIABLE}"][:]
        lat_idx, lon_idx = np.unravel_index(
            np.argmin(np.abs(lats - lat) + np.abs(lons - lon)),
            lats.shape
        )
        return {"Date": date_str, "Satellite": float(data[lat_idx, lon_idx])}

# -------------------- LOAD FIELD DATA --------------------
print("Loading field data from Excel...")
df_field = pd.read_excel(EXCEL_FILENAME)
df_field = df_field.rename(columns={TIMESTAMP_COLUMN: "Date", FIELD_COLUMN_NAME: "Field"})
df_field["Date"] = pd.to_datetime(df_field["Date"])
df_field = df_field.drop_duplicates(subset="Date").sort_values("Date")

# Build a full 30-min time base for the period
start_dt = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE)
full_timebase = pd.date_range(start=start_dt, end=end_dt + timedelta(days=1), freq="30min")[:-1]
df = pd.DataFrame({"Date": full_timebase})
df = pd.merge(df, df_field, on="Date", how="left")

# -------------------- DOWNLOAD SMAP DATA --------------------
print("Downloading SMAP data...")
smap_records = []
current_dt = start_dt
while current_dt <= end_dt:
    date_str = current_dt.strftime("%Y-%m-%d")
    try:
        url = find_smap_file(date_str)
        if url:
            fname = url.split("/")[-1]
            print(f"  {date_str} -> {fname}")
            if download_smap_file(url, fname):
                entry = extract_smap_value(fname, LATITUDE, LONGITUDE, date_str)
                smap_records.append(entry)
                os.remove(fname)
            else:
                print(f"  Failed to download SMAP for {date_str}")
        else:
            print(f"  No SMAP file found for {date_str}")
    except Exception as e:
        print(f"  ERROR on {date_str}: {e}")
    current_dt += timedelta(days=1)

df_smap = pd.DataFrame(smap_records)
df_smap["Date"] = pd.to_datetime(df_smap["Date"])
df_smap = df_smap.set_index("Date").resample("30min").ffill().reset_index()
df = pd.merge(df, df_smap, on="Date", how="left")

# -------------------- ALIGN SATELLITE TO FIELD --------------------
# Calculate the mean offset on days when both field and satellite data are available
common = df.dropna(subset=["Field", "Satellite"])
if not common.empty:
    offset = common["Field"].mean() - common["Satellite"].mean()
    df["Satellite_Aligned"] = df["Satellite"] + offset
else:
    df["Satellite_Aligned"] = df["Satellite"]

# -------------------- FINAL INTERPOLATION --------------------
# For any missing field values, we take the aligned satellite value,
# then use linear interpolation to fill any remaining gaps.
df["Final"] = df["Field"]
mask = df["Field"].isna() & df["Satellite_Aligned"].notna()
df.loc[mask, "Final"] = df.loc[mask, "Satellite_Aligned"]
df["Final"] = df["Final"].interpolate("linear", limit_direction="both")

# -------------------- OUTPUT ONLY DATE AND FINAL --------------------
final_output = df[["Date", "Final"]].copy()
final_output.to_csv(FINAL_CSV, index=False)
print(f"Final CSV saved to {FINAL_CSV}")

# -------------------- PLOT THE FINAL RESULT --------------------
plt.figure(figsize=(16,6))
plt.plot(df["Date"], df["Final"], label="Final Interpolated", color="green")
plt.plot(df["Date"], df["Satellite_Aligned"], label="Aligned SMAP", color="orange", linestyle="--")
plt.scatter(df["Date"], df["Field"], label="Raw Field", s=5, color="gray", alpha=0.6)
plt.title("Final Soil Moisture — Field + SMAP Fusion")
plt.xlabel("Date")
plt.ylabel("Soil Moisture (m³/m³)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
