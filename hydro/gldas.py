import os
import requests
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ==== USER SETTINGS ====
USERNAME = "your_earthdata_username"
PASSWORD = "your_earthdata_password"
LAT = 30.27
LON = -97.74 
START_DATE = datetime(2021, 2, 1)
END_DATE = datetime(2022, 2, 2)
FIELD_CSV = "sample_field_et.csv"

# ==== INIT: DELETE OLD .nc4 FILES ====
for file in os.listdir():
    if file.endswith(".nc4"):
        os.remove(file)
        print(f"Deleted old file: {file}")

# ==== DOWNLOAD + EXTRACT GLDAS ET ====
results = []
date = START_DATE
while date <= END_DATE:
    doy = date.timetuple().tm_yday
    yyyy = date.year
    yyyymmdd = date.strftime('%Y%m%d')

    for hour in range(0, 24, 3):
        hhmm = f"{hour:02d}00"
        fname = f"GLDAS_NOAH025_3H.A{yyyymmdd}.{hhmm}.021.nc4"
        url = f"https://data.gesdisc.earthdata.nasa.gov/data/GLDAS/GLDAS_NOAH025_3H.2.1/{yyyy}/{doy:03d}/{fname}"

        print(f"Downloading: {fname}")
        try:
            with requests.Session() as session:
                session.auth = (USERNAME, PASSWORD)
                r = session.get(url, stream=True)
                if r.status_code != 200:
                    print(f"  Failed: {r.status_code}")
                    continue
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)

            # Extract ET from .nc4
            ds = xr.open_dataset(fname)
            et_var = next((v for v in ["Evap_tavg", "evap_tavg", "ET", "et", "evap", "Evap"]), None)
            if et_var not in ds:
                print(f"  No ET var in {fname}")
                ds.close()
                os.remove(fname)
                continue

            et = ds[et_var].sel(lat=LAT, lon=LON, method='nearest')
            et_mm = float(et) * 10800  # Convert kg/m²/s to mm
            timestamp = datetime.strptime(f"{yyyymmdd}{hhmm}", "%Y%m%d%H%M")
            results.append({"datetime": timestamp, "et_mm": et_mm})

            ds.close()
            os.remove(fname)
            print(f"  Processed + deleted: {fname}")

        except Exception as e:
            print(f"  Error: {e}")
            if os.path.exists(fname):
                os.remove(fname)

    date += timedelta(days=1)

# ==== BUILD 30-MIN TIME SERIES ====
df_gldas = pd.DataFrame(results).sort_values("datetime")
if df_gldas.empty:
    print("No GLDAS data downloaded. Exiting.")
    exit()

timebase = pd.date_range(start=df_gldas["datetime"].min(), end=df_gldas["datetime"].max(), freq="30min")
df = pd.DataFrame({"datetime": timebase})
df = pd.merge(df, df_gldas, on="datetime", how="left")
df["et_mm"] = df["et_mm"].ffill()

# ==== LOAD FIELD DATA FROM EXCEL ====
excel_file = "DATA.xlsx"
df_field = pd.read_excel(excel_file)

# Convert timestamp column to datetime and rename for merging
df_field["datetime"] = pd.to_datetime(df_field["TIMESTAMP"])
df_field = df_field.rename(columns={"ET": "field_et"})


df = pd.merge(df, df_field[["datetime", "field_et"]], on="datetime", how="left")


# ==== ALIGN + FUSE ====
common = df.dropna(subset=["et_mm", "field_et"])
offset = common["field_et"].mean() - common["et_mm"].mean() if not common.empty else 0
df["gldas_aligned"] = df["et_mm"] + offset

df["Final_ET"] = df["field_et"]
mask = df["field_et"].isna() & df["gldas_aligned"].notna()
df.loc[mask, "Final_ET"] = df.loc[mask, "gldas_aligned"]

# ==== INTERPOLATE + FILTER ====
df["Final_ET"] = df["Final_ET"].interpolate("linear", limit_direction="both")
drop_mask = df["Final_ET"].diff() < -0.05
df.loc[drop_mask, "Final_ET"] = np.nan
df["Final_ET"] = df["Final_ET"].interpolate("linear", limit_direction="both")

# ==== SAVE + PLOT ====
df[["datetime", "Final_ET"]].to_csv("gldas_et_fused_output.csv", index=False)
print("Saved: gldas_et_fused_output.csv")

plt.figure(figsize=(14,6))
plt.plot(df["datetime"], df["Final_ET"], label="Final Fused ET", color="green")
plt.plot(df["datetime"], df["gldas_aligned"], label="Aligned GLDAS ET", linestyle="--", color="orange")
plt.scatter(df["datetime"], df["field_et"], label="Field ET", color="black", s=40, alpha=0.6)
plt.title("GLDAS-Fused Evapotranspiration (30-min)")
plt.xlabel("Time")
plt.ylabel("ET (mm)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
