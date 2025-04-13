import os
import requests
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# -------------------- USER CONFIG --------------------
LATITUDE = 30.5
LONGITUDE = -96.5
START_DATE = "2018-04-01"
END_DATE = "2018-05-15"

EXCEL_FILE = "RealData.xlsx"
TIMESTAMP_COLUMN = "Timestamp"
FIELD_COLUMN = "Precip1_tot"
FIELD_WEIGHT = 0.7

# -------------------- IMERG URL --------------------
def generate_imerg_url(date):
    ymd = date.strftime("%Y%m%d")
    y, m = date.strftime("%Y"), date.strftime("%m")
    fname = f"3B-DAY.MS.MRG.3IMERG.{ymd}-S000000-E235959.V07B.nc4"
    url = f"https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/{y}/{m}/{fname}"
    return url, fname

# -------------------- DOWNLOAD FILE --------------------
def download_file(url, filename):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    return False

# -------------------- EXTRACT IMERG VALUE --------------------
def extract_precip(filename, lat, lon, date):
    with h5py.File(filename, "r") as f:
        latitudes = f["lat"][:]
        longitudes = f["lon"][:]
        precip = f["precipitation"][:]  # (1, lat, lon)
        lat_idx = np.argmin(np.abs(latitudes - lat))
        lon_idx = np.argmin(np.abs(longitudes - lon))
        return {"Date": date, "IMERG": float(precip[0, lat_idx, lon_idx])}

# -------------------- LOAD FIELD DATA --------------------
def load_field_data(filepath):
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    if TIMESTAMP_COLUMN not in df.columns or FIELD_COLUMN not in df.columns:
        raise ValueError(f"Missing columns: {TIMESTAMP_COLUMN} or {FIELD_COLUMN}")
    df["Date"] = pd.to_datetime(df[TIMESTAMP_COLUMN]).dt.floor("D")
    df["Field"] = pd.to_numeric(df[FIELD_COLUMN], errors="coerce")
    daily_df = df.groupby("Date")["Field"].sum().reset_index()
    print(f"🧪 Field data range: {daily_df['Date'].min().date()} to {daily_df['Date'].max().date()}")
    return daily_df

# -------------------- BLEND FIELD + SAT --------------------
def blend_field_sat(sat_df, field_df):
    full_range = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    df = pd.DataFrame({"Date": full_range})
    df = df.merge(sat_df, on="Date", how="left")
    df = df.merge(field_df, on="Date", how="left")

    df["Blended"] = df["Field"]
    for i in range(len(df)):
        if pd.isna(df.iloc[i]["Field"]):
            field_interp = df["Field"].interpolate("linear").iloc[i]
            sat_val = df.iloc[i]["IMERG"]
            if not pd.isna(sat_val):
                blended = FIELD_WEIGHT * field_interp + (1 - FIELD_WEIGHT) * sat_val
            else:
                blended = field_interp
            df.iloc[i, df.columns.get_loc("Blended")] = blended

    df["Blended"] = df["Blended"].interpolate("linear", limit_direction="both")

    print("\n📊 Final Blending Overview:")
    print(df[["Date", "Field", "IMERG", "Blended"]].to_string(index=False))

    return df

# -------------------- MAIN --------------------
print("📥 Downloading IMERG data...")
sat_data = []
current = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE)

while current <= end_dt:
    url, fname = generate_imerg_url(current)
    print(f"→ {current.date()}: ", end="")
    try:
        if download_file(url, fname):
            row = extract_precip(fname, LATITUDE, LONGITUDE, current)
            sat_data.append(row)
            os.remove(fname)
            print(f"✓ {row['IMERG']:.2f} mm")
        else:
            print("✘ Failed")
    except Exception as e:
        print(f"⚠ ERROR: {e}")
    current += timedelta(days=1)

sat_df = pd.DataFrame(sat_data)
sat_df["Date"] = pd.to_datetime(sat_df["Date"])
field_df = load_field_data(EXCEL_FILE)
final_df = blend_field_sat(sat_df, field_df)

# -------------------- SAVE & BAR PLOT --------------------
final_df.to_csv("Final_Reconstructed_Precip.csv", index=False)
print("✔ Saved Final_Reconstructed_Precip.csv")

plt.figure(figsize=(14, 6))

# Bar width and x-offsets
bar_width = 0.25
dates = final_df["Date"]
x = np.arange(len(dates))

# Bars for each source
plt.bar(x - bar_width, final_df["Field"], width=bar_width, label="Field", color="black")
plt.bar(x, final_df["Blended"], width=bar_width, label="Interpolated", color="skyblue")
plt.bar(x + bar_width, final_df["IMERG"], width=bar_width, label="IMERG", color="orange")

# X-axis ticks
plt.xticks(ticks=x[::max(1, len(x)//15)], labels=dates.dt.strftime('%Y-%m-%d')[::max(1, len(x)//15)], rotation=45)

plt.title("Daily Precipitation — IMERG + Field Fusion (All Bars)")
plt.xlabel("Date")
plt.ylabel("Precipitation (mm/day)")
plt.legend()
plt.tight_layout()
plt.grid(True, axis='y')
plt.show()
