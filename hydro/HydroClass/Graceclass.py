import subprocess
import sys
import os

class GraceFieldBlender:
    def __init__(self):
        pass

   

    # ----------------------------------------
    # Download GRACE NetCDF
    # ----------------------------------------
    def download_nc_file(self, username, password, url, output_file):
        import requests
        print("Downloading GRACE NetCDF...")
        try:
            with requests.get(url, auth=(username, password), stream=True) as r:
                r.raise_for_status()
                with open(output_file, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
            print(f"Downloaded: {output_file}")
        except Exception as e:
            print(f"Download failed: {e}")

    # ----------------------------------------
    # Extract GRACE Satellite Data
    # ----------------------------------------
    def load_satellite_data(self, nc_file, start_year, end_year, latitude, longitude):
        import netCDF4 as nc
        import numpy as np
        from datetime import datetime
        print("Loading satellite data...")
        ds = nc.Dataset(nc_file, 'r')
        lats = ds.variables['lat'][:]
        lons = ds.variables['lon'][:]
        time = ds.variables['time'][:]
        data = ds.variables['lwe_thickness'][:]
        dates = nc.num2date(time, units=ds.variables['time'].units)
        dates = [datetime(d.year, d.month, d.day) for d in dates]

        lat_idx = np.argmin(np.abs(lats - latitude))
        lon_idx = np.argmin(np.abs(lons - longitude))

        values, final_dates = [], []
        for i, d in enumerate(dates):
            if start_year <= d.year <= end_year:
                values.append(data[i, lat_idx, lon_idx])
                final_dates.append(d)

        import pandas as pd
        sat_df = pd.DataFrame({"Date": final_dates, "Satellite_Anomaly": values})
        return sat_df

    # ----------------------------------------
    # Load Field Data (User Column)
    # ----------------------------------------
    def load_field_data(self, filepath, date_col, value_col):
        import pandas as pd
        print("Loading field data...")
        df = pd.read_excel(filepath)
        df[date_col] = pd.to_datetime(df[date_col])
        return df[[date_col, value_col]].rename(columns={date_col: "Date", value_col: "Field_Measurement"})

    # ----------------------------------------
    # Normalize Satellite to Field
    # ----------------------------------------
    def normalize_satellite_to_field(self, sat_df, field_df):
        print("Normalizing satellite data...")
        field_avg = (field_df["Field_Measurement"].max() + field_df["Field_Measurement"].min()) / 2
        sat_avg = (sat_df["Satellite_Anomaly"].max() + sat_df["Satellite_Anomaly"].min()) / 2
        sat_df["Satellite_Anomaly"] += (field_avg - sat_avg)
        return sat_df

    # ----------------------------------------
    # Merge & Interpolate with Weighting
    # ----------------------------------------
    def process_data_weighted(self, sat_df, field_df, mode="field_only", beta=2, epsilon=0.1, static_weight=0.7):
        import numpy as np
        import pandas as pd
        print("Processing with weighting mode:", mode)
        df = pd.merge(sat_df, field_df, on="Date", how="outer").sort_values("Date")

        df["SD_sat"] = df["Satellite_Anomaly"].rolling(7, center=True, min_periods=1).std()
        df["SD_field"] = df["Field_Measurement"].rolling(7, center=True, min_periods=1).std()
        df["Interpolated_Value"] = df["Field_Measurement"]

        for i in range(len(df)):
            field = df.iloc[i]["Field_Measurement"]
            sat = df.iloc[i]["Satellite_Anomaly"]

            if pd.isna(field):
                interp_field = df["Field_Measurement"].interpolate().iloc[i]

                if mode == "static":
                    w = static_weight
                elif mode == "dynamic":
                    s_sd = df.iloc[i]["SD_sat"]
                    f_sd = df.iloc[i]["SD_field"]
                    if pd.isna(s_sd) or pd.isna(f_sd) or f_sd + epsilon == 0:
                        w = 0.5
                    else:
                        ratio = (s_sd / (f_sd + epsilon)) ** beta
                        w = 1 / (1 + ratio)
                else:  # field_only
                    w = 1 if not pd.isna(interp_field) else 0

                if not pd.isna(sat):
                    blended = w * interp_field + (1 - w) * sat
                else:
                    blended = interp_field

                df.iloc[i, df.columns.get_loc("Interpolated_Value")] = blended

        df["Interpolated_Value"] = df["Interpolated_Value"].interpolate(limit_direction="both")
        return df

    # ----------------------------------------
    # Save to Final CSV
    # ----------------------------------------
    def save_final_csv(self, df, output_file="Final_Reconstructed_Time_Series.csv"):
        import pandas as pd
        print("Saving final CSV...")
        full_range = pd.date_range(start=df["Date"].min(), end=df["Date"].max(), freq="D")
        output = pd.DataFrame({"Date": full_range})
        merged = df[["Date", "Field_Measurement", "Satellite_Anomaly", "Interpolated_Value"]]
        output = output.merge(merged, on="Date", how="left")
        output["Interpolated_Value"] = output["Interpolated_Value"].interpolate(limit_direction="both")
        output.to_csv(output_file, index=False)
        print(f"{output_file} saved")

    # ----------------------------------------
    # Plot Result
    # ----------------------------------------
    def plot_all(self, df):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(df["Date"], df["Interpolated_Value"], label="Interpolated", color="green", linewidth=2)
        if "Satellite_Anomaly" in df.columns:
            plt.plot(df["Date"], df["Satellite_Anomaly"], label="Satellite", color="orange", linestyle="--")
        if "Field_Measurement" in df.columns:
            mask = df["Field_Measurement"].notna()
            plt.scatter(df.loc[mask, "Date"], df.loc[mask, "Field_Measurement"], color="black", label="Field Data", s=15)
        plt.title("GRACE-Aligned Groundwater Time Series")
        plt.xlabel("Date")
        plt.ylabel("Groundwater Depth (m)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ----------------------------------------
    # Clean Extra Files
    # ----------------------------------------
    def cleanup(self, keep=["Final_Reconstructed_Time_Series.csv", "RealData.xlsx"]):
        print("Cleaning up...")
        for f in os.listdir():
            if (f.endswith(".csv") or f.endswith(".nc")) and f not in keep:
                try:
                    os.remove(f)
                    print(f"Deleted: {f}")
                except Exception as e:
                    print(f"Could not delete {f}: {e}")

    # ----------------------------------------
    # Run the full workflow
    # ----------------------------------------
    def run(self, username, password, nc_url, nc_file,
            excel_file, date_column, value_column,
            start_year, end_year, latitude, longitude,
            weighting_mode="dynamic", beta=2, epsilon=0.1, static_weight=0.7,
            output_file="Final_Reconstructed_Time_Series.csv", cleanup_keep=None):
        if cleanup_keep is None:
            cleanup_keep = [output_file, excel_file]
        self.install_packages()
        self.download_nc_file(username, password, nc_url, nc_file)
        sat_df = self.load_satellite_data(nc_file, start_year, end_year, latitude, longitude)
        field_df = self.load_field_data(excel_file, date_column, value_column)
        sat_df = self.normalize_satellite_to_field(sat_df, field_df)
        combined = self.process_data_weighted(sat_df, field_df, mode=weighting_mode, beta=beta, epsilon=epsilon, static_weight=static_weight)
        self.save_final_csv(combined, output_file=output_file)
        self.plot_all(combined)
        self.cleanup(keep=cleanup_keep)



'''
#HOW TO USE
blender = GraceFieldBlender()
blender.run(
    username="your_username",
    password="your_password",
    nc_url = "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4/GRCTellus.JPL.200204_202411.GLO.RL06.3M.MSCNv04CRI.nc" #get rid
    nc_file="GRCTellus_data.nc",
    excel_file="RealData.xlsx",
    date_column="Date",
    value_column="GW depth,in m, at Soil node 2",
    start_year=2019,
    end_year=2021,
    latitude=28 + 46 / 60 + 49.8 / 3600,
    longitude=-(95 + 36 / 60 + 51.7 / 3600),
    weighting_mode="dynamic",  # or "static", "field_only"
    beta=2,
    epsilon=0.1,
    static_weight=0.7,
    output_file="Final_Reconstructed_Time_Series.csv"
)

'''