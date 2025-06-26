import os
import sys
import subprocess
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
from datetime import datetime, timedelta

class SatelliteFusionProcessor:
    def __init__(self,
                 username,
                 password,
                 nc_url,
                 nc_file,
                 excel_file,
                 date_column,
                 value_column,
                 start_year,
                 end_year,
                 latitude,
                 longitude,
                 weighting_mode="dynamic",
                 beta=2,
                 epsilon=0.1,
                 static_weight=0.7):
        self.username = username
        self.password = password
        self.nc_url = nc_url
        self.nc_file = nc_file
        self.excel_file = excel_file
        self.date_column = date_column
        self.value_column = value_column
        self.start_year = start_year
        self.end_year = end_year
        self.latitude = latitude
        self.longitude = longitude
        self.weighting_mode = weighting_mode
        self.beta = beta
        self.epsilon = epsilon
        self.static_weight = static_weight

        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def install_packages(self):
        required = ["requests", "pandas", "numpy", "matplotlib", "netCDF4", "openpyxl", "xarray"]
        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                print(f"Installing: {pkg}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

    # =============== GRACE PIPELINE ===============
    def download_grace_data(self):
        print("Downloading GRACE NetCDF...")
        try:
            with requests.get(self.nc_url, auth=(self.username, self.password), stream=True) as r:
                r.raise_for_status()
                with open(self.nc_file, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
            print(f"Downloaded: {self.nc_file}")
        except Exception as e:
            print(f"Download failed: {e}")

    def load_grace_satellite_data(self):
        print("Loading satellite data...")
        ds = nc.Dataset(self.nc_file, 'r')
        lats = ds.variables['lat'][:]
        lons = ds.variables['lon'][:]
        time = ds.variables['time'][:]
        data = ds.variables['lwe_thickness'][:]
        dates = nc.num2date(time, units=ds.variables['time'].units)
        dates = [datetime(d.year, d.month, d.day) for d in dates]

        lat_idx = np.argmin(np.abs(lats - self.latitude))
        lon_idx = np.argmin(np.abs(lons - self.longitude))

        values, final_dates = [], []
        for i, d in enumerate(dates):
            if self.start_year <= d.year <= self.end_year:
                values.append(data[i, lat_idx, lon_idx])
                final_dates.append(d)

        return pd.DataFrame({"Date": final_dates, "Satellite_Anomaly": values})

    def load_field_data(self):
        print("Loading field data...")
        df = pd.read_excel(self.excel_file)
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        return df[[self.date_column, self.value_column]].rename(columns={
            self.date_column: "Date",
            self.value_column: "Field_Measurement"
        })

    def normalize_satellite_to_field(self, sat_df, field_df):
        print("Normalizing satellite data...")
        field_avg = (field_df["Field_Measurement"].max() + field_df["Field_Measurement"].min()) / 2
        sat_avg = (sat_df["Satellite_Anomaly"].max() + sat_df["Satellite_Anomaly"].min()) / 2
        sat_df["Satellite_Anomaly"] += (field_avg - sat_avg)
        return sat_df

    def process_data_weighted(self, sat_df, field_df):
        print("Processing with weighting mode:", self.weighting_mode)
        df = pd.merge(sat_df, field_df, on="Date", how="outer").sort_values("Date")

        df["SD_sat"] = df["Satellite_Anomaly"].rolling(7, center=True, min_periods=1).std()
        df["SD_field"] = df["Field_Measurement"].rolling(7, center=True, min_periods=1).std()
        df["Interpolated_Value"] = df["Field_Measurement"]

        for i in range(len(df)):
            field = df.iloc[i]["Field_Measurement"]
            sat = df.iloc[i]["Satellite_Anomaly"]

            if pd.isna(field):
                interp_field = df["Field_Measurement"].interpolate().iloc[i]

                if self.weighting_mode == "static":
                    w = self.static_weight
                elif self.weighting_mode == "dynamic":
                    s_sd = df.iloc[i]["SD_sat"]
                    f_sd = df.iloc[i]["SD_field"]
                    if pd.isna(s_sd) or pd.isna(f_sd) or f_sd + self.epsilon == 0:
                        w = 0.5
                    else:
                        ratio = (s_sd / (f_sd + self.epsilon)) ** self.beta
                        w = 1 / (1 + ratio)
                else:
                    w = 1 if not pd.isna(interp_field) else 0

                if not pd.isna(sat):
                    blended = w * interp_field + (1 - w) * sat
                else:
                    blended = interp_field

                df.iloc[i, df.columns.get_loc("Interpolated_Value")] = blended

        df["Interpolated_Value"] = df["Interpolated_Value"].interpolate(limit_direction="both")
        return df

    def save_final_csv(self, df, filename="Final_Reconstructed_Time_Series.csv"):
        print("Saving final CSV...")
        full_range = pd.date_range(start=df["Date"].min(), end=df["Date"].max(), freq="D")
        output = pd.DataFrame({"Date": full_range})
        merged = df[["Date", "Field_Measurement", "Satellite_Anomaly", "Interpolated_Value"]]
        output = output.merge(merged, on="Date", how="left")
        output["Interpolated_Value"] = output["Interpolated_Value"].interpolate(limit_direction="both")
        output.to_csv(filename, index=False)
        print(f"{filename} saved")

    def plot_all(self, df):
        plt.figure(figsize=(12, 6))
        plt.plot(df["Date"], df["Interpolated_Value"], label="Interpolated", color="green", linewidth=2)
        if "Satellite_Anomaly" in df.columns:
            plt.plot(df["Date"], df["Satellite_Anomaly"], label="Satellite", color="orange", linestyle="--")
        if "Field_Measurement" in df.columns:
            mask = df["Field_Measurement"].notna()
            plt.scatter(df.loc[mask, "Date"], df.loc[mask, "Field_Measurement"], color="black", label="Field Data", s=15)
        plt.title("GRACE-Unified Time Series")
        plt.xlabel("Date")
        plt.ylabel("Groundwater Depth (m)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def cleanup(self, keep=["Final_Reconstructed_Time_Series.csv", "RealData.xlsx"]):
        print("Cleaning up...")
        for f in os.listdir():
            if (f.endswith(".csv") or f.endswith(".nc")) and f not in keep:
                try:
                    os.remove(f)
                    print(f"Deleted: {f}")
                except Exception as e:
                    print(f"Could not delete {f}: {e}")

    def run_grace_pipeline(self):
        self.download_grace_data()
        sat_df = self.load_grace_satellite_data()
        field_df = self.load_field_data()
        sat_df = self.normalize_satellite_to_field(sat_df, field_df)
        combined_df = self.process_data_weighted(sat_df, field_df)
        self.save_final_csv(combined_df)
        self.plot_all(combined_df)
        self.cleanup()

    # =============== GLDAS PIPELINE ===============
    def run_gldas_pipeline(self,
                           gldas_username,
                           gldas_password,
                           lat,
                           lon,
                           start_date,
                           end_date,
                           field_excel,
                           field_timestamp_col,
                           field_et_col,
                           out_csv="gldas_et_fused_output.csv"):
        self.cleanup_gldas_nc4()
        gldas_df = self.download_extract_gldas_et(
            gldas_username, gldas_password, lat, lon, start_date, end_date
        )
        if gldas_df.empty:
            print("No GLDAS data downloaded. Exiting.")
            return
        df_out = self.fuse_gldas_with_field(
            gldas_df, field_excel, field_timestamp_col, field_et_col
        )
        df_out[["datetime", "Final_ET"]].to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
        self.plot_gldas_et_fusion(df_out)

    def cleanup_gldas_nc4(self):
        for file in os.listdir():
            if file.endswith(".nc4"):
                try:
                    os.remove(file)
                    print(f"Deleted old file: {file}")
                except Exception as e:
                    print(f"Error deleting file {file}: {e}")

    def download_extract_gldas_et(self, username, password, lat, lon, start_date, end_date):
        results = []
        date = start_date
        while date <= end_date:
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
                        session.auth = (username, password)
                        r = session.get(url, stream=True)
                        if r.status_code != 200:
                            print(f"  Failed: {r.status_code}")
                            continue
                        with open(fname, "wb") as f:
                            for chunk in r.iter_content(8192):
                                f.write(chunk)
                    # Extract ET
                    ds = xr.open_dataset(fname)
                    et_var = None
                    for v in ["Evap_tavg", "evap_tavg", "ET", "et", "evap", "Evap"]:
                        if v in ds.variables:
                            et_var = v
                            break
                    if not et_var:
                        print(f"  No ET var in {fname}")
                        ds.close()
                        os.remove(fname)
                        continue
                    et = ds[et_var].sel(lat=lat, lon=lon, method='nearest')
                    et_mm = float(et.values) * 10800  # Convert kg/m²/s to mm (3 hours = 10800s)
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
        df_gldas = pd.DataFrame(results).sort_values("datetime")
        return df_gldas

    def fuse_gldas_with_field(self, df_gldas, excel_file, timestamp_col, et_col):
        # Build 30-min timebase
        timebase = pd.date_range(start=df_gldas["datetime"].min(), end=df_gldas["datetime"].max(), freq="30min")
        df = pd.DataFrame({"datetime": timebase})
        df = pd.merge(df, df_gldas, on="datetime", how="left")
        df["et_mm"] = df["et_mm"].ffill()
        # Load field data
        df_field = pd.read_excel(excel_file)
        df_field["datetime"] = pd.to_datetime(df_field[timestamp_col])
        df_field = df_field.rename(columns={et_col: "field_et"})
        df = pd.merge(df, df_field[["datetime", "field_et"]], on="datetime", how="left")
        # Align & fuse
        common = df.dropna(subset=["et_mm", "field_et"])
        offset = common["field_et"].mean() - common["et_mm"].mean() if not common.empty else 0
        df["gldas_aligned"] = df["et_mm"] + offset
        df["Final_ET"] = df["field_et"]
        mask = df["field_et"].isna() & df["gldas_aligned"].notna()
        df.loc[mask, "Final_ET"] = df.loc[mask, "gldas_aligned"]
        # Interpolate & drop correction
        df["Final_ET"] = df["Final_ET"].interpolate("linear", limit_direction="both")
        drop_mask = df["Final_ET"].diff() < -0.05
        df.loc[drop_mask, "Final_ET"] = np.nan
        df["Final_ET"] = df["Final_ET"].interpolate("linear", limit_direction="both")
        return df

    def plot_gldas_et_fusion(self, df):
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
        
        
# =============== IMERG PIPELINE ===============
    def run_imerg_pipeline(self,
                          latitude,
                          longitude,
                          start_date,
                          end_date,
                          excel_file,
                          timestamp_column,
                          field_column,
                          field_weight=0.7,
                          weight_mode="static",  # options: "field_only", "static", "dynamic"
                          beta=2,
                          epsilon=0.1,
                          out_csv="Final_Reconstructed_Precip.csv"):
        print(" Downloading IMERG data...")
        sat_data = []
        current = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        while current <= end_dt:
            url, fname = self.generate_imerg_url(current)
            print(f"→ {current.date()}: ", end="")
            try:
                if self.download_file(url, fname):
                    row = self.extract_imerg_precip(fname, latitude, longitude, current)
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
        field_df = self.load_field_precip_data(excel_file, timestamp_column, field_column)
        final_df = self.blend_field_sat(sat_df, field_df, start_date, end_date,
                                        weight_mode, field_weight, beta, epsilon)
        final_df.to_csv(out_csv, index=False)
        print(f"✔ Saved {out_csv}")
        self.plot_imerg_precip(final_df)

    def generate_imerg_url(self, date):
        ymd = date.strftime("%Y%m%d")
        y, m = date.strftime("%Y"), date.strftime("%m")
        fname = f"3B-DAY.MS.MRG.3IMERG.{ymd}-S000000-E235959.V07B.nc4"
        url = f"https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/{y}/{m}/{fname}"
        return url, fname

    def download_file(self, url, filename):
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
        return False

    def extract_imerg_precip(self, filename, lat, lon, date):
        with h5py.File(filename, "r") as f:
            latitudes = f["lat"][:]
            longitudes = f["lon"][:]
            precip = f["precipitation"][:]  # (1, lat, lon)
            lat_idx = np.argmin(np.abs(latitudes - lat))
            lon_idx = np.argmin(np.abs(longitudes - lon))
            return {"Date": date, "IMERG": float(precip[0, lat_idx, lon_idx])}

    def load_field_precip_data(self, filepath, timestamp_column, field_column):
        df = pd.read_excel(filepath)
        df.columns = df.columns.str.strip()
        if timestamp_column not in df.columns or field_column not in df.columns:
            raise ValueError(f"Missing columns: {timestamp_column} or {field_column}")
        df["Date"] = pd.to_datetime(df[timestamp_column]).dt.floor("D")
        df["Field"] = pd.to_numeric(df[field_column], errors="coerce")
        daily_df = df.groupby("Date")["Field"].sum().reset_index()
        print(f"Field data range: {daily_df['Date'].min().date()} to {daily_df['Date'].max().date()}")
        return daily_df

    def blend_field_sat(self, sat_df, field_df, start_date, end_date,
                        weight_mode="field_only", field_weight=0.7, beta=2, epsilon=0.1):
        full_range = pd.date_range(start=start_date, end=end_date, freq="D")
        df = pd.DataFrame({"Date": full_range})
        df = df.merge(sat_df, on="Date", how="left")
        df = df.merge(field_df, on="Date", how="left")

        df["SD_sat"] = df["IMERG"].rolling(window=7, center=True, min_periods=1).std()
        df["SD_field"] = df["Field"].rolling(window=7, center=True, min_periods=1).std()
        df["Blended"] = df["Field"]

        for i in range(len(df)):
            field_val = df.iloc[i]["Field"]
            sat_val = df.iloc[i]["IMERG"]

            if pd.isna(field_val):
                field_interp = df["Field"].interpolate("linear").iloc[i]

                if weight_mode == "static":
                    w = field_weight
                elif weight_mode == "dynamic":
                    sd_sat = df.iloc[i]["SD_sat"]
                    sd_field = df.iloc[i]["SD_field"]
                    if pd.isna(sd_sat) or pd.isna(sd_field) or sd_field + epsilon == 0:
                        w = 0.5
                    else:
                        ratio = (sd_sat / (sd_field + epsilon)) ** beta
                        w = 1 / (1 + ratio)
                else:  # "field_only"
                    w = 1 if not pd.isna(field_interp) else 0

                if not pd.isna(sat_val):
                    blended = w * field_interp + (1 - w) * sat_val
                else:
                    blended = field_interp

                df.iloc[i, df.columns.get_loc("Blended")] = blended

        df["Blended"] = df["Blended"].interpolate("linear", limit_direction="both")
        return df

    def plot_imerg_precip(self, final_df):
        plt.figure(figsize=(14, 6))
        bar_width = 0.25
        dates = final_df["Date"]
        x = np.arange(len(dates))
        plt.bar(x - bar_width, final_df["Field"], width=bar_width, label="Field", color="black")
        plt.bar(x, final_df["Blended"], width=bar_width, label="Interpolated", color="skyblue")
        plt.bar(x + bar_width, final_df["IMERG"], width=bar_width, label="IMERG", color="orange")
        plt.xticks(ticks=x[::max(1, len(x)//15)],
                   labels=dates.dt.strftime('%Y-%m-%d')[::max(1, len(x)//15)], rotation=45)
        plt.title("Daily Precipitation (IMERG) Unified timeseries")
        plt.xlabel("Date")
        plt.ylabel("Precipitation (mm/day)")
        plt.legend()
        plt.tight_layout()
        plt.grid(True, axis='y')
        plt.show()
