import os
import requests
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class IMERGFieldBlender:
    def __init__(self):
        pass

    # -------------------- GENERATE IMERG URL --------------------
    def generate_imerg_url(self, date):
        ymd = date.strftime("%Y%m%d")
        y, m = date.strftime("%Y"), date.strftime("%m")
        fname = f"3B-DAY.MS.MRG.3IMERG.{ymd}-S000000-E235959.V07B.nc4"
        url = f"https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/{y}/{m}/{fname}"
        return url, fname

    # -------------------- DOWNLOAD FILE --------------------
    def download_file(self, url, filename):
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
        return False

    # -------------------- EXTRACT IMERG VALUE --------------------
    def extract_precip(self, filename, lat, lon, date):
        with h5py.File(filename, "r") as f:
            latitudes = f["lat"][:]
            longitudes = f["lon"][:]
            precip = f["precipitation"][:]  # (1, lat, lon)
            lat_idx = np.argmin(np.abs(latitudes - lat))
            lon_idx = np.argmin(np.abs(longitudes - lon))
            return {"Date": date, "IMERG": float(precip[0, lat_idx, lon_idx])}

    # -------------------- LOAD FIELD DATA --------------------
    def load_field_data(self, filepath, timestamp_column, field_column):
        df = pd.read_excel(filepath)
        df.columns = df.columns.str.strip()
        if timestamp_column not in df.columns or field_column not in df.columns:
            raise ValueError(f"Missing columns: {timestamp_column} or {field_column}")
        df["Date"] = pd.to_datetime(df[timestamp_column]).dt.floor("D")
        df["Field"] = pd.to_numeric(df[field_column], errors="coerce")
        daily_df = df.groupby("Date")["Field"].sum().reset_index()
        print(f"🧪 Field data range: {daily_df['Date'].min().date()} to {daily_df['Date'].max().date()}")
        return daily_df

    # -------------------- BLEND FIELD + SAT --------------------
    def blend_field_sat(self, sat_df, field_df, start_date, end_date, weight_mode="field_only", field_weight=0.7, beta=2, epsilon=0.1):
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

        print("\n📊 Final Blending Overview:")
        print(df[["Date", "Field", "IMERG", "Blended"]].to_string(index=False))

        return df

    # -------------------- SAVE TO CSV --------------------
    def save_to_csv(self, df, filename="Final_Reconstructed_Precip.csv"):
        df.to_csv(filename, index=False)
        print(f"✔ Saved {filename}")

    # -------------------- BAR PLOT --------------------
    def bar_plot(self, df):
        plt.figure(figsize=(14, 6))
        bar_width = 0.25
        dates = df["Date"]
        x = np.arange(len(dates))

        plt.bar(x - bar_width, df["Field"], width=bar_width, label="Field", color="black")
        plt.bar(x, df["Blended"], width=bar_width, label="Interpolated", color="skyblue")
        plt.bar(x + bar_width, df["IMERG"], width=bar_width, label="IMERG", color="orange")

        plt.xticks(ticks=x[::max(1, len(x)//15)], labels=dates.dt.strftime('%Y-%m-%d')[::max(1, len(x)//15)], rotation=45)
        plt.title("Daily Precipitation — IMERG + Field Fusion (All Bars)")
        plt.xlabel("Date")
        plt.ylabel("Precipitation (mm/day)")
        plt.legend()
        plt.tight_layout()
        plt.grid(True, axis='y')
        plt.show()

    # -------------------- RUN THE WHOLE PIPELINE --------------------
    def run(self,
            latitude,
            longitude,
            start_date,
            end_date,
            excel_file,
            timestamp_column,
            field_column,
            field_weight=0.7,
            weight_mode="dynamic",
            beta=2,
            epsilon=0.1,
            save_csv="Final_Reconstructed_Precip.csv"):

        print("📥 Downloading IMERG data...")
        sat_data = []
        current = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        while current <= end_dt:
            url, fname = self.generate_imerg_url(current)
            print(f"→ {current.date()}: ", end="")
            try:
                if self.download_file(url, fname):
                    row = self.extract_precip(fname, latitude, longitude, current)
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
        field_df = self.load_field_data(excel_file, timestamp_column, field_column)
        final_df = self.blend_field_sat(
            sat_df,
            field_df,
            start_date=start_date,
            end_date=end_date,
            weight_mode=weight_mode,
            field_weight=field_weight,
            beta=beta,
            epsilon=epsilon
        )
        self.save_to_csv(final_df, filename=save_csv)
        self.bar_plot(final_df)


'''
#how to use

blender = IMERGFieldBlender()
blender.run(
    latitude=30.5,
    longitude=-96.5,
    start_date="2018-04-01",
    end_date="2018-05-15",
    excel_file="RealData.xlsx",
    timestamp_column="Timestamp",
    field_column="Precip1_tot",
    field_weight=0.7,
    weight_mode="dynamic",  # "static", "field_only", or "dynamic"
    beta=2,
    epsilon=0.1,
    save_csv="Final_Reconstructed_Precip.csv"
)


'''