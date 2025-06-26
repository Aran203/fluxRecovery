import os
import h5py
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class SMAPFieldBlender:
    def __init__(self):
        pass

    # ------------- NASA CMR API: Find File -------------
    def find_smap_file(self, date_str, cmr_api_url="https://cmr.earthdata.nasa.gov/search/granules.json"):
        params = {
            "short_name": "SPL4SMGP",
            "version": "007",
            "temporal": f"{date_str}T00:00:00Z,{date_str}T23:59:59Z",
            "page_size": 1,
            "sort_key": "-start_date",
        }
        r = requests.get(cmr_api_url, params=params)
        if r.status_code == 200:
            entries = r.json().get("feed", {}).get("entry", [])
            if entries:
                return entries[0]["links"][0]["href"]
        return None

    # ------------- Download File -------------
    def download_smap_file(self, url, filename, username, password):
        r = requests.get(url, stream=True, auth=(username, password))
        if r.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        return False

    # ------------- Extract SMAP Value -------------
    def extract_smap_value(self, fname, lat, lon, smap_variable, date_str):
        with h5py.File(fname, "r") as f:
            lats = f["cell_lat"][:]
            lons = f["cell_lon"][:]
            data = f[f"Geophysical_Data/{smap_variable}"][:]
            lat_idx, lon_idx = np.unravel_index(
                np.argmin(np.abs(lats - lat) + np.abs(lons - lon)),
                lats.shape
            )
            return {"Date": date_str, "Satellite": float(data[lat_idx, lon_idx])}

    # ------------- Load Field Data -------------
    def load_field_data(self, excel_filename, timestamp_column, field_column_name):
        print("Loading field data from Excel...")
        df_field = pd.read_excel(excel_filename)
        df_field = df_field.rename(columns={timestamp_column: "Date", field_column_name: "Field"})
        df_field["Date"] = pd.to_datetime(df_field["Date"])
        df_field = df_field.drop_duplicates(subset="Date").sort_values("Date")
        return df_field

    # ------------- Main Pipeline -------------
    def run(self,
            latitude,
            longitude,
            start_date,
            end_date,
            username,
            password,
            excel_filename,
            timestamp_column,
            field_column_name,
            depth_cm,
            final_csv="Final_Reconstructed_Time_Series.csv",
            plot_title=None):

        # -------- Determine SMAP Variable --------
        if depth_cm <= 5:
            smap_variable = "sm_surface"
            sat_layer_desc = "Surface (0-5 cm)"
        elif depth_cm >= 15:
            smap_variable = "sm_rootzone"
            sat_layer_desc = "Root Zone (~0-100 cm)"
        else:
            smap_variable = "sm_surface"
            sat_layer_desc = "Surface fallback (no exact match)"
            print(f"Note: No exact SMAP match for {depth_cm} cm. Using surface as fallback.")

        # --------- Load Field Data ---------
        df_field = self.load_field_data(excel_filename, timestamp_column, field_column_name)

        # Build a full 30-min time base for the period
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        full_timebase = pd.date_range(start=start_dt, end=end_dt + timedelta(days=1), freq="30min")[:-1]
        df = pd.DataFrame({"Date": full_timebase})
        df = pd.merge(df, df_field, on="Date", how="left")

        # --------- Download & Extract SMAP Data ---------
        print("Downloading SMAP data...")
        smap_records = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_str = current_dt.strftime("%Y-%m-%d")
            try:
                url = self.find_smap_file(date_str)
                if url:
                    fname = url.split("/")[-1]
                    print(f"  {date_str} -> {fname}")
                    if self.download_smap_file(url, fname, username, password):
                        entry = self.extract_smap_value(fname, latitude, longitude, smap_variable, date_str)
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

        # --------- Align Satellite to Field ---------
        common = df.dropna(subset=["Field", "Satellite"])
        if not common.empty:
            offset = common["Field"].mean() - common["Satellite"].mean()
            df["Satellite_Aligned"] = df["Satellite"] + offset
        else:
            df["Satellite_Aligned"] = df["Satellite"]

        # --------- Final Interpolation ---------
        df["Final"] = df["Field"]
        mask = df["Field"].isna() & df["Satellite_Aligned"].notna()
        df.loc[mask, "Final"] = df.loc[mask, "Satellite_Aligned"]
        df["Final"] = df["Final"].interpolate("linear", limit_direction="both")

        # --------- Save Final Output ---------
        final_output = df[["Date", "Final"]].copy()
        final_output.to_csv(final_csv, index=False)
        print(f"Final CSV saved to {final_csv}")

        # --------- Plot Result ---------
        plt.figure(figsize=(16, 6))
        plt.plot(df["Date"], df["Final"], label="Final Interpolated", color="green")
        plt.plot(df["Date"], df["Satellite_Aligned"], label="Aligned SMAP", color="orange", linestyle="--")
        plt.scatter(df["Date"], df["Field"], label="Raw Field", s=5, color="gray", alpha=0.6)
        if not plot_title:
            plot_title = "Final Soil Moisture — Field + SMAP Fusion"
        plt.title(plot_title)
        plt.xlabel("Date")
        plt.ylabel("Soil Moisture (m³/m³)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


'''
#how to use
blender = SMAPFieldBlender()
blender.run(
    latitude=30.5,
    longitude=-96.5,
    start_date="2018-01-01",
    end_date="2018-12-31",
    username="your_earthdata_username",
    password="your_earthdata_password",
    excel_filename="RealData.xlsx",
    timestamp_column="Timestamp",
    field_column_name="VWC_1_Avg",
    depth_cm=5,  # or 15, etc.
    final_csv="Final_Reconstructed_Time_Series.csv"
)


'''