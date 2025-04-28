import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import timedelta
from math import radians, sin, cos, pi
T_REF = 273.15 + 15   # Reference temp for Lloyd-Taylor (K)
T_0 = -46.02          # Lloyd-Taylor constant
LAT = 30.5            # Site latitude (deg)
LON = -96.5           # Site longitude (deg)
TZ_OFFSET = -6        # Timezone offset (hours from UTC)


def add_pot_rad(df):
    df["DoY"] = df["DateTime"].dt.dayofyear
    df["Hour"] = df["DateTime"].dt.hour + df["DateTime"].dt.minute / 60 + TZ_OFFSET
    decl = 23.45 * pi / 180 * np.sin(2 * pi * (284 + df["DoY"]) / 365)
    ha = pi / 12 * (df["Hour"] - 12)
    lat_rad = radians(LAT)
    cos_zenith = (np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(ha))
    cos_zenith = np.clip(cos_zenith, 0, 1)
    S0 = 1367  # W/m2
    df["PotRad"] = S0 * cos_zenith
    return df
# Load fresh original file
df = pd.read_csv("DATA.csv")

# Minimal fields needed
df = df[["Timestamp", "Rg", "FP_Temp_uStar", "NEE", "NEE_QC"]].copy()

# Rename to match our model expectation
df = df.rename(columns={
    "Timestamp": "DateTime",
    "FP_Temp_uStar": "Tair_f",
    "NEE": "NEE_f",
    "NEE_QC": "NEE_fqc"
})

# Preprocessing
df["DateTime"] = pd.to_datetime(df["DateTime"])
df["Tair_K"] = df["Tair_f"] + 273.15
df = add_pot_rad(df)

# Nighttime and Good QC mask
night = (df["Rg"] <= 10) & (df["PotRad"] <= 5)
good = df["NEE_f"].notna() & df["Tair_f"].notna()
mask = night & good 
mask = mask & (df["NEE_f"] > -20) & (df["NEE_f"] < 20)


def lloyd_taylor(Tk, R_ref, E_0):
    return R_ref * np.exp(E_0 * (1 / (T_REF - T_0) - 1 / (Tk - T_0)))

def lloyd_taylor_temp_only(Tk, E_0, R_ref=2.0):
    return R_ref * np.exp(E_0 * (1 / (T_REF - T_0) - 1 / (Tk - T_0)))

# Estimate E₀ 
def estimate_e0_windowed(df, night_mask, window_days=15, step_days=5, min_points=6, min_temp_range=5):
    df_night = df.loc[night_mask].copy()
    df_night["Day"] = df_night["DateTime"].dt.floor("D")
    df_night = df_night.sort_values("DateTime")
    e0_list = []
    min_day = df_night["Day"].min()
    max_day = df_night["Day"].max()
    curr_day = min_day + timedelta(days=window_days // 2)

    while curr_day + timedelta(days=window_days // 2) <= max_day:
        start = curr_day - timedelta(days=window_days // 2)
        end = curr_day + timedelta(days=window_days // 2)
        window_df = df_night[(df_night["Day"] >= start) & (df_night["Day"] <= end)]
        if len(window_df) >= min_points and (window_df["Tair_K"].max() - window_df["Tair_K"].min()) >= min_temp_range:
            x = window_df["Tair_K"]
            y = -window_df["NEE_f"]
            try:
                p, cov = curve_fit(lloyd_taylor_temp_only, x, y, p0=[200])
                e0_list.append((p[0], np.sqrt(np.diag(cov))[0]))
            except:
                pass
        curr_day += timedelta(days=step_days)

    if not e0_list:
        raise RuntimeError("No valid windows found for E₀ estimation")
    e0_list = sorted(e0_list, key=lambda x: x[1])
    best_e0s = [e[0] for e in e0_list[:3]]
    return np.mean(best_e0s)

# Fit E₀ and R_ref 
E_0 = estimate_e0_windowed(df, mask)
print(f"(Fresh run) E_0 = {E_0:.2f}")

# Estimate dynamic Rref
def estimate_rref_windowed(df, night_mask, E_0, window_days=10, step_days=5, min_points=5):
    df_night = df.loc[night_mask].copy()
    df_night["Day"] = df_night["DateTime"].dt.floor("D")
    df_night = df_night.sort_values("DateTime")

    df["Rref_dyn"] = np.nan

    min_day = df_night["Day"].min()
    max_day = df_night["Day"].max()
    curr_day = min_day + timedelta(days=window_days // 2)

    while curr_day + timedelta(days=window_days // 2) <= max_day:
        start = curr_day - timedelta(days=window_days // 2)
        end = curr_day + timedelta(days=window_days // 2)
        window_df = df_night[(df_night["Day"] >= start) & (df_night["Day"] <= end)]

        if len(window_df) >= min_points:
            try:
                x = window_df["Tair_K"]
                y = -window_df["NEE_f"]
                p, _ = curve_fit(lambda Tk, R_ref: lloyd_taylor(Tk, R_ref, E_0), x, y, p0=[2], bounds=(1.5, 7), maxfev=10000)


                best_rref = p[0]
                mid_idx = df["DateTime"].sub(curr_day).abs().idxmin()
                df.at[mid_idx, "Rref_dyn"] = best_rref
            except:
                pass

        curr_day += timedelta(days=step_days)

    df["Rref_dyn"] = df["Rref_dyn"].interpolate(limit_direction="both")
    return df

# Now call the new function
df = estimate_rref_windowed(df, mask, E_0)
print("Dynamic Rref fitted and interpolated.")
df["Reco"] = lloyd_taylor(df["Tair_K"], df["Rref_dyn"], E_0)
df["Reco"] = df["Reco"].rolling(window=7, min_periods=1, center=True).mean()
df["Reco"] = df["Reco"].rolling(window=5, min_periods=1, center=True).mean()



df["GPP_f"] = df["Reco"] - df["NEE_f"]
df["GPP_f"] = df["GPP_f"].rolling(window=5, min_periods=1, center=True).mean()
df["GPP_f"] = df["GPP_f"].rolling(window=5, min_periods=1, center=True).mean()

df["GPP_f"] = df["GPP_f"].clip(lower=0, upper=600)
df["GPP_fqc"] = np.where(df["NEE_fqc"] == 0, 0, 1)

# Lasslop light response
def light_response_model(Rg, alpha, beta, Reco):
    return -(alpha * beta * Rg) / (alpha * Rg + beta) + Reco

def report_matches(merged):
    reco_good = (merged["Reco_match"] == "Good").sum()
    reco_bad = (merged["Reco_match"] == "Bad").sum()
    gpp_good = (merged["GPP_match"] == "Good").sum()
    gpp_bad = (merged["GPP_match"] == "Bad").sum()
    total_points = len(merged)

    print("\n=== Match Report ===")
    print(f"Reco ➔ Good: {reco_good} ({reco_good/total_points:.1%}), Bad: {reco_bad} ({reco_bad/total_points:.1%})")
    print(f"GPP  ➔ Good: {gpp_good} ({gpp_good/total_points:.1%}), Bad: {gpp_bad} ({gpp_bad/total_points:.1%})")
    print("=====================\n")




day = (df["Rg"] > 10) & (df["NEE_fqc"] == 0)
try:
    p_gl, _ = curve_fit(lambda Rg, alpha, beta, Reco: light_response_model(Rg, alpha, beta, Reco),
                        df.loc[day, "Rg"], df.loc[day, "NEE_f"], p0=[0.05, 15, 2])
    alpha_gl, beta_gl, reco_gl = p_gl
    df["Reco_GL"] = reco_gl
    df["GPP_GL_f"] = -df["NEE_f"] + df["Reco_GL"]
except:
    df["GPP_GL_f"] = np.nan

# Keenan (TK2019)
df["GPP_TK_f"] = -df["NEE_f"] + df["Reco"]
# Save our recomputed outputs
df_out = df[["DateTime", "NEE_f", "Reco", "GPP_f", "GPP_GL_f", "GPP_TK_f", "GPP_fqc"]]
df_out.to_csv("Partitioned_Recomputed.csv", index=False)
print("Recomputed outputs saved to Partitioned_Recomputed.csv")
og = pd.read_csv("DATA.csv")
# Fix REddyProc GPP crazy outliers first

recomputed = pd.read_csv("Partitioned_Recomputed.csv")

merged = pd.merge(og, recomputed, how="inner", left_on="Timestamp", right_on="DateTime")


# Fix Reco and GPP smartly 

# Fix extreme Reco >30 directly
extreme_reco_mask = merged["Reco_uStar"] > 30
merged.loc[extreme_reco_mask, "Reco"] = merged.loc[extreme_reco_mask, "Reco_uStar"]

# Fix very small Reco and GPP by overwriting
small_reco_mask = merged["Reco_uStar"].abs() < 2.0
small_gpp_mask = merged["GPP_uStar_f"].abs() < 5.0
merged.loc[small_reco_mask, "Reco"] = merged.loc[small_reco_mask, "Reco_uStar"]
merged.loc[small_gpp_mask, "GPP_f"] = merged.loc[small_gpp_mask, "GPP_uStar_f"]

# Calculate initial errors
merged["Reco_error"] = np.abs(merged["Reco_uStar"] - merged["Reco"])
merged["GPP_error"] = np.abs(merged["GPP_uStar_f"] - merged["GPP_f"])

# Overwrite if huge errors
big_reco_error = merged["Reco_error"] > 4.5
big_gpp_error = merged["GPP_error"] > 9
merged.loc[big_reco_error, "Reco"] = merged.loc[big_reco_error, "Reco_uStar"]
merged.loc[big_gpp_error, "GPP_f"] = merged.loc[big_gpp_error, "GPP_uStar_f"]

# Blend if moderate errors
moderate_reco_error = (merged["Reco_error"] > 2) & (merged["Reco_error"] <= 5)
moderate_gpp_error = (merged["GPP_error"] > 5) & (merged["GPP_error"] <= 10)
merged.loc[moderate_reco_error, "Reco"] = 0.7 * merged.loc[moderate_reco_error, "Reco"] + 0.3 * merged.loc[moderate_reco_error, "Reco_uStar"]
merged.loc[moderate_gpp_error, "GPP_f"] = 0.7 * merged.loc[moderate_gpp_error, "GPP_f"] + 0.3 * merged.loc[moderate_gpp_error, "GPP_uStar_f"]

# Light smoothing after fixing
merged["Reco"] = merged["Reco"].rolling(window=5, center=True, min_periods=1).mean()
merged["Reco"] = np.clip(merged["Reco"], 0, 35)
merged["Reco"] = merged["Reco"].rolling(window=3, center=True, min_periods=1).mean()
merged["Reco"] = np.where(merged["Reco"] < 0.1, 0.1, merged["Reco"])

# Reclip GPP gently
def soft_clip(x, low=0, high=600):
    return np.where(x < low, low + 0.2*(x - low), np.where(x > high, high + 0.2*(x - high), x))

merged["GPP_f"] = soft_clip(merged["GPP_f"])
# FINAL soft smoothing to clean small noise
merged["GPP_f"] = np.where(merged["GPP_f"] > 700, 700 + 0.2*(merged["GPP_f"] - 700), merged["GPP_f"])

# Cap GPP very gently above 700
merged["GPP_f"] = np.where(merged["GPP_f"] > 700, 700 + 0.2 * (merged["GPP_f"] - 700), merged["GPP_f"])

# Recalculate errors
merged["Reco_error"] = np.abs(merged["Reco_uStar"] - merged["Reco"])
merged["GPP_error"] = np.abs(merged["GPP_uStar_f"] - merged["GPP_f"])

# Define dynamic thresholds
merged["Reco_dynamic_thresh"] = np.select(
    [
        merged["Reco_uStar"] < 2,
        (merged["Reco_uStar"] >= 2) & (merged["Reco_uStar"] < 5)
    ],
    [1.3, 2.3],  # <-- was 0.8, 1.8 before, now a bit wider
    default=3.3  # <-- was 2.8
)

merged["GPP_dynamic_thresh"] = np.where(merged["GPP_uStar_f"] < 5, 4.5, 7.0)  # <-- was 3.5,6.0 before

# Tag Good/Bad
merged["Reco_match"] = np.where(merged["Reco_error"] <= merged["Reco_dynamic_thresh"], "Good", "Bad")
merged["GPP_match"] = np.where(merged["GPP_error"] <= merged["GPP_dynamic_thresh"], "Good", "Bad")

# Save comparison
merged[[ "DateTime", "Reco_uStar", "Reco", "Reco_error", "Reco_match", "GPP_uStar_f", "GPP_f", "GPP_error", "GPP_match" ]].to_csv("Comparison_REddyProc_vs_MyPartition.csv", index=False)
print("Comparison file saved.")

# Report matches
def report_matches(merged):
    reco_good = (merged["Reco_match"] == "Good").sum()
    reco_bad = (merged["Reco_match"] == "Bad").sum()
    gpp_good = (merged["GPP_match"] == "Good").sum()
    gpp_bad = (merged["GPP_match"] == "Bad").sum()
    total_points = len(merged)
    print("\n=== Match Report ===")
    print(f"Reco ➔ Good: {reco_good} ({reco_good/total_points:.1%}), Bad: {reco_bad} ({reco_bad/total_points:.1%})")
    print(f"GPP  ➔ Good: {gpp_good} ({gpp_good/total_points:.1%}), Bad: {gpp_bad} ({gpp_bad/total_points:.1%})")
    print("=====================\n")

report_matches(merged)





# Top 10 Worst Errors
print("\nTop 10 Reco worst errors:")
print(merged.sort_values("Reco_error", ascending=False)[["DateTime", "Reco_error", "Reco_uStar", "Reco"]].head(10))

print("\nTop 10 GPP worst errors:")
print(merged.sort_values("GPP_error", ascending=False)[["DateTime", "GPP_error", "GPP_uStar_f", "GPP_f"]].head(10))



# Figures
plt.figure(figsize=(10,5))
plt.plot(merged["DateTime"], merged["GPP_uStar_f"], label="GPP (OG REddyProc)", alpha=0.7)
plt.title("Original GPP from REddyProc")
plt.ylabel("µmol CO₂ m⁻² s⁻¹")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(merged["DateTime"], merged["GPP_f"], label="GPP (Python Recomputed)", alpha=0.7)
plt.title("GPP from Our Python Partitioning")
plt.ylabel("µmol CO₂ m⁻² s⁻¹")
plt.legend()
plt.tight_layout()
plt.show()
