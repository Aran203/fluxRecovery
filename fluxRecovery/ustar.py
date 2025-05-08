import pandas as pd
import numpy as np

class fluxRecovery:

    def __init__(self, data):
        self.EddyData = data.copy()
        
        for column in ["Ustar", "NEE", "VPD", "Tair", "Rg", "DateTime"]:
            assert(column in self.EddyData.columns)

        self.EddyData["DateTime"] = data["DateTime"] - pd.to_timedelta(15, unit = 'm')

        self.ustar_results = None
        self.sTEMP = None
        self.DTS = 48               # TODO: number of dailytime time stamps   CHANGE LATER
        
    def create_season_factor_month(self, dates, start_months=[3, 6, 9, 12]):
        """
        Compute year-spanning season factor by starting month.

        Parameters:
            dates (pd.Series): A pandas Series of datetime objects.
            start_months (list): List of integers specifying the starting months for each season.

        Returns:
            pd.Series: A factor (categorical) indicating the season for each date.
        """
        
        dates = pd.to_datetime(dates)
        months = dates.dt.month
        years = dates.dt.year

        if (len(years) != len(months)):
            raise ValueError("Lengths of years and months do not match")

        start_months = sorted(set(start_months))
        unique_years = sorted(years.unique())
        start_year_months = np.array([y * 1000 + m for y in unique_years for m in start_months])

        if start_months[0] != 1:
            year_begin = start_year_months[0] // 1000
            start_year_months = np.insert(start_year_months, 0, 1000*year_begin + 1)
        
        
        year_months = years * 1000 + months
        season_factor = np.zeros_like(year_months, dtype=int)
        
        for i in range(len(start_year_months) - 1):
            mask = (year_months >= start_year_months[i]) & (year_months < start_year_months[i + 1])
            season_factor[mask] = start_year_months[i]
        
        season_factor[year_months >= start_year_months[-1]] = start_year_months[-1]
        
        return pd.Series(season_factor).astype(str)

    def get_year_of_season(self, season_factor, dates):
        """
        Determine the year of the record corresponding to the middle of each season.

        Parameters:
            season_factor (pd.Series): A categorical Series indicating the season for each record.
            dates (pd.Series): A pandas Series of datetime objects.

        Returns:
            dict: A mapping of season to the corresponding year.
        """
        
        years = np.ones(dates.shape[0], dtype = int)

        for season in season_factor.unique():

            mask = season_factor == season 
            mask = mask[mask]

            start_idx = mask.index[0]
            last_idx = mask.index[-1]

            mid_idx = (start_idx + last_idx) // 2
            yr_mid = dates.iloc[mid_idx].year


            years[start_idx:last_idx+1] = yr_mid
        
        return years

    def get_valid_ustar_indices(self, data, sw_thr=10):
        """
        Remove non-finite cases and constrain to nighttime data.

        Parameters:
            data (pd.DataFrame): Input data with columns for Ustar, NEE, Tair, and Rg.
            ustar_col (str): Column name for Ustar.
            nee_col (str): Column name for NEE.
            temp_col (str): Column name for air temperature.
            rg_col (str): Column name for solar radiation.
            sw_thr (float): Threshold below which data is considered nighttime.

        Returns:
            pd.Series: A boolean Series indicating valid records.
        """

        
        # Check if values in each column are finite (similar to R's is.finite())
        valid_ustar = np.isfinite(data["Ustar"])
        valid_nee = np.isfinite(data["NEE"])
        valid_temp = np.isfinite(data["Tair"])
        valid_rg = np.isfinite(data["Rg"])
        
        # Combine all checks to create a valid mask
        valid = valid_ustar & valid_nee & valid_temp & valid_rg
        
        # Apply the nighttime condition (solar radiation below threshold)
        valid = valid & (data["Rg"] < sw_thr)

        return valid

    def bin_data(self, data, col, n_bins):
        """
        Bin data into equal-sized bins.

        Parameters:
            data (pd.Series): The data to be binned.
            col (str): Column name to bin.
            n_bins (int): Number of bins.

        Returns:
            pd.Series: A Series indicating the bin for each record.
        """
        

        # Sort the data by the column to be binned
        sorted_data = data.sort_values(by = col).reset_index()
        x = sorted_data[col].values

        # Calculate the bin size
        bin_size = len(x) // n_bins
        bin_ids = np.zeros(len(x), dtype=int)

        # Assign bins based on intervals
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else len(x)
            bin_ids[start:end] = i + 1

        # Return the bin IDs as a Series, mapped back to the original index
        return pd.Series(bin_ids, index=sorted_data["index"], name="bin")

    def bin_ustar(self, data, n_bins=20):
        """
        Bin NEE and Ustar into Ustar classes.

        Parameters:
            data (pd.DataFrame): Input data with columns for NEE and Ustar.
            nee_col (str): Column name for NEE.
            ustar_col (str): Column name for Ustar.
            n_bins (int): Number of Ustar bins.

        Returns:
            pd.DataFrame: DataFrame with mean NEE and Ustar for each bin.
        """
        

        sorted_data = data.sort_values(by="Ustar").reset_index()
        x = sorted_data["Ustar"].values

        # Calculate the bin size
        bin_size = len(x) // n_bins
        bin_ids = np.zeros(len(x), dtype=int)

        # Assign bins based on intervals
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else len(x)
            bin_ids[start:end] = i + 1

        # Add the bin IDs to the sorted data
        sorted_data["ustar_bin"] = bin_ids

        # Map the 'ustar_bin' column back to the original data
        data = data.merge(
            sorted_data[["index", "ustar_bin"]], left_index=True, right_on="index", how="left"
        )
        data.drop(columns=["index"], inplace=True)

        # Calculate bin means
        bin_means = sorted_data.groupby("ustar_bin").agg(
            ustar_mean=("Ustar", "mean"),
            nee_mean=("NEE", "mean")
        ).reset_index()

        return bin_means, data

    def estimate_ustar_threshold_single(self, bin_means, plateau_crit=0.95, forward_bins=10, min_plateau_bins=3):
        """
        Estimate the Ustar threshold for a single subset using a stricter moving point method.

        Parameters:
            bin_means (pd.DataFrame): DataFrame with mean NEE and Ustar for each bin.
            plateau_crit (float): Criterion for detecting a plateau (default: 0.95).
            forward_bins (int): Number of forward bins to compare.
            min_plateau_bins (int): Minimum number of bins required for a valid plateau (default: 3).

        Returns:
            float: The estimated Ustar threshold.
        """

        for i in range(len(bin_means) - min_plateau_bins):
            # Current bin NEE
            current_nee = bin_means["nee_mean"].iloc[i]
            next_nee = bin_means["nee_mean"].iloc[i + 1]

            # Forward bins' mean NEE
            forward_nee = bin_means["nee_mean"].iloc[i + 1 : i + 1 + forward_bins].mean()
            next_forward_nee = bin_means["nee_mean"].iloc[i + 2 : i + 2 + forward_bins].mean()

            # Stricter plateau criterion: both current and next bins must satisfy the condition
            if (current_nee >= plateau_crit * forward_nee and next_nee >= plateau_crit * next_forward_nee):
                # Ensure there are enough bins in the plateau
                if len(bin_means) - i >= min_plateau_bins:
                    return bin_means["ustar_mean"].iloc[i]

        # If not satisfied, fallback is the ustar value of the last possible bin check 
        # return bin_means["ustar_mean"].iloc[forward_bins+1]
        return np.nan


    def estimate_ustar_threshold(self, data):
        """
        Main workflow for estimating Ustar thresholds for each temp class for a given dataset.

        Parameters:
            data (pd.DataFrame): Input data with columns for date, temperature, ustar, and NEE.
    
        Returns:
            A vector of #Temp classes of Ustar Thresholds
            pd.DataFrame: Seasonal and yearly Ustar thresholds.
        """

        TEMP_CLASSES = 7
        MIN_RECORDS_IN_PERIOD = 160
        MIN_RECORDS_IN_TEMP_CLASS = 100

        data = data.copy()
        records = data.shape[0]
        # print(data.head())

        if (records < MIN_RECORDS_IN_PERIOD or (records / TEMP_CLASSES) < MIN_RECORDS_IN_TEMP_CLASS):
            # print("Not enough rows for dataset")
            
            return [np.nan for i in range(TEMP_CLASSES)]


        ustar_thresholds = [np.nan for i in range(TEMP_CLASSES)]

        data['temp_bin'] = self.bin_data(data, "Tair", n_bins=TEMP_CLASSES)

        for temp_bin in range(1, TEMP_CLASSES+1):
            temp_data = data[data['temp_bin'] == temp_bin]
            temp_data = temp_data.sort_values(by="Ustar")

            
            bin_means, data_modified = self.bin_ustar(temp_data, n_bins=20)

            if (bin_means['ustar_mean'].iloc[0] > 0.2):
                continue
        
            threshold = self.estimate_ustar_threshold_single(bin_means)

            ustar_thresholds[temp_bin - 1] = threshold 

        return ustar_thresholds


    def compute_ustar_values(self, data, inBootstrap = False):
        '''Assumes that data has seasonfactor and year column'''

        MIN_SAMPLES_YR = 3000


        valid_indices = self.get_valid_ustar_indices(data)
        dc = data[valid_indices].reset_index(drop = True)                 #data clean


        all_seasons = data.groupby("season").agg(
            year = ("year", lambda x: x.mode()[0])
        ).reset_index()

        valid_records_season = dc.groupby("season").agg(
            samples = ("season", "size"),
            year = ("year", lambda x: x.mode()[0])
        ).reset_index()


        all_seasons = pd.merge(all_seasons, valid_records_season, how = "outer", on = ['season', 'year'])
        all_years = all_seasons.groupby("year")['samples'].sum().reset_index(name = "samples")

        years_low_samples = set(all_years[all_years['samples'] < MIN_SAMPLES_YR]['year'])
        
        season_rsts = {}
        for season in all_seasons['season'].unique():
            season_thresholds = self.estimate_ustar_threshold(dc[dc['season'] == season])
            season_rsts[season] = season_thresholds 

        # df with seasons (COLUMNS) x TEMPCLASSES associated USTAR VALUES
        season_rsts = pd.DataFrame(season_rsts)

        results_seasons = all_seasons.copy(deep=True)
        results_seasons['uStarEst'] = results_seasons['season'].map(season_rsts.median(skipna=True))

        results_yrs = results_seasons.groupby("year").agg(
            samples = ("samples", "sum"), 
            uStarMaxSeason = ("uStarEst", "max")
        ).reset_index()
        results_yrs['uStarAgg'] = results_yrs["uStarMaxSeason"]


        yrs_pooled = set(results_yrs[(results_yrs['samples'] > 0) & (results_yrs['uStarAgg'].isna())]['year'])
        
        if (True):   # CHANGE LATER
            yrs_pooled = yrs_pooled.union(years_low_samples)


        if (len(yrs_pooled)):
            dc_pooled = dc[dc["year"].isin(yrs_pooled)]

            yr_pooled_rsts = {}
            for yr in dc_pooled['year'].unique():
                yr_thresholds = self.estimate_ustar_threshold(dc_pooled[dc_pooled['year'] == yr])
                yr_pooled_rsts[yr] = yr_thresholds 

            yr_pooled_rsts = pd.DataFrame(yr_pooled_rsts)

            results_pooled = results_yrs[results_yrs["year"].isin(yrs_pooled)][["year", "samples"]]
            results_pooled["uStarPooled"] = results_pooled["year"].map(yr_pooled_rsts.median(skipna=True))
        else: 
            results_pooled = pd.DataFrame(np.nan, index = [0], columns = ["year", "samples", "uStarPooled"])
        
        
        results_pooled = results_pooled.dropna(how="all")        
        results_yrs = pd.merge(results_yrs, results_pooled, how = "outer", on = ["year", "samples"])    

    
        is_finite = np.isfinite(results_yrs["uStarPooled"])
        results_yrs.loc[is_finite, "uStarAgg"] = results_yrs.loc[is_finite, "uStarPooled"]  


        median_yrs = results_yrs["uStarAgg"].median(skipna=True)
        non_finite = ~np.isfinite(results_yrs["uStarAgg"])
        results_yrs.loc[non_finite, "uStarAgg"] = median_yrs



        results_seasons = pd.merge(results_seasons, results_yrs[["year", "uStarAgg"]], on = "year", how = "outer")
        is_finite_est = np.isfinite(results_seasons["uStarEst"])
        results_seasons.loc[is_finite_est, "uStarAgg"] = results_seasons.loc[is_finite_est, "uStarEst"]


        season_results_df = results_seasons[["season", "year", "uStarAgg"]].copy()
        season_results_df["aggregation_mode"] = "season"


        yrs_results_temp = results_yrs[["year", "uStarAgg"]].copy()
        yrs_results_temp["aggregation_mode"] = "year"
        yrs_results_temp["season"] = "NA"


        results_all = pd.concat([yrs_results_temp, season_results_df], ignore_index = True)

        temp_df = pd.DataFrame({"aggregation_mode": ["single"], "season": ["NA"], "year": ["NA"], "uStarAgg": [median_yrs]})
        results_all = pd.concat([temp_df, results_all], ignore_index = True)

        results_all = results_all.rename(columns = {"uStarAgg": "uStar"})

        return {"thresholds":  results_all, "thresholds_yrs": results_yrs, "thresholds_seasons": results_seasons, "ustar_dense": season_rsts}


    def compute_ustar_scenarios(self, samples = 10, quantiles = [0.05, 0.5, 0.95]):
        '''computes baseline ustar scnearios and does bootstrapping'''

        ds = self.EddyData

        ds['season'] = self.create_season_factor_month(ds["DateTime"])
        ds['year'] = self.get_year_of_season(ds['season'], ds["DateTime"])

        res_base = self.compute_ustar_values(ds)
        res_base = res_base['thresholds']

        years = res_base[res_base['aggregation_mode'] == "year"]["year"]
        seasons = res_base[res_base['aggregation_mode'] == "season"]["season"]


        agg = res_base[res_base['aggregation_mode'] == "single"]["uStar"]
        agg.index = ["agg"] * len(agg)

        yrs = res_base[res_base['aggregation_mode'] == "year"]["uStar"]
        yrs.index = years

        sns = res_base[res_base['aggregation_mode'] == "season"]["uStar"]
        sns.index = seasons 


        ustar_values = pd.concat([agg, yrs, sns])

        
        def helper(seed):
            np.random.seed(seed)

            ds_boot = ds.groupby("season", group_keys = False).apply(lambda group: group.sample(n = len(group), replace = True))

            res_boot = self.compute_ustar_values(ds_boot) 
            res_boot = res_boot["thresholds"]

            yr_mask = res_boot['aggregation_mode'] == "year"
            season_mask = res_boot['aggregation_mode'] == "season"

            years_boot = res_boot[yr_mask]["year"]
            seasons_boot = res_boot[season_mask]["season"]


            res_agg_boot = res_boot[res_boot["aggregation_mode"] == "single"]["uStar"]
            res_agg_boot.index = ["agg"] * len(res_agg_boot)

            if np.array_equal(years_boot, years):
                res_yrs_boot = res_boot[yr_mask]["uStar"]
                res_yrs_boot.index = years  
            else:
                res_yrs_boot = pd.Series(np.full(len(years.shape[0]), np.nan), index = years)
            

            if res_boot.shape[0] == res_base.shape[0] and np.array_equal(seasons_boot, seasons):
                res_seasons_boot = res_boot[season_mask]["uStar"]
                res_seasons_boot.index = seasons 
            else:
                res_seasons_boot = pd.Series(np.full(len(seasons.shape[0]), np.nan), index = seasons)


            res_boot = pd.concat([res_agg_boot, res_yrs_boot, res_seasons_boot])

            return res_boot


        seeds = np.random.randint(1, np.iinfo(np.int32).max, size = samples - 1)

        responses_boot = [ustar_values]
        for seed in seeds:
            responses_boot.append(helper(seed))


        boot_results_df = pd.DataFrame([s for s in responses_boot])
        boot_results_df.columns = responses_boot[0].index  


        quantiles_df = boot_results_df.quantile(quantiles)
        quantiles_df = quantiles_df.T 

        cols = ["uStar" + str(int(float(i)*100)) for i in quantiles_df.columns]
        quantiles_df.columns = cols

        res_df = pd.concat([res_base.reset_index(drop = True), quantiles_df.reset_index(drop = True)], axis = 1)
        
        self.ustar_results = res_df
        self.ustar_mapp = self.get_annual_season_map(self.ustar_results)


    # COMMENT OUT LATER?
    def set_ustar_mapp(self, df):
        self.ustar_mapp = df
        
    def get_annual_season_map(self, df):
        # print(df)
        seasons = df.loc[df['aggregation_mode'] == 'season'][['season', 'year']]

        years = df.loc[df['aggregation_mode'] == "year"]
        years = years.drop(columns = ['aggregation_mode', 'season'])

        mapp = pd.merge(seasons, years, on = ['year'])
        mapp = mapp.drop(columns = ['year'])
        
        return mapp
    
    def get_seasonal_map(self, df):
        
        mapp = df.loc[df['aggregation_mode'] == 'season']
        mapp = mapp.drop(columns = ['aggregation_mode', 'year'])
        return mapp


    def fSetQF(self, data, var, qf_var, qf_value):
        if qf_var != 'none':
            return data[var].where(data[qf_var] == qf_value, np.nan)
        else:
            return data[var].copy()


    def TEMP_init(self, var, qf_var, qf_value, fill_all = True):

        filtered_data = self.fSetQF(self.EddyData.merge(self.sTEMP), var, qf_var, qf_value)
        nrows = filtered_data.shape[0]

        temp = pd.DataFrame({
            "VAR_orig": filtered_data, 
            "VAR_f" : filtered_data,
            "VAR_fqc": [np.nan for _ in range(nrows)],
            "VAR_fall": [np.nan for _ in range(nrows)],
            "VAR_fall_qc": [np.nan for _ in range(nrows)],
            "VAR_fnum": [np.nan for _ in range(nrows)],
            "VAR_fsd": [np.nan for _ in range(nrows)],
            "VAR_fmeth": [np.nan for _ in range(nrows)],
            "VAR_fwin": [np.nan for _ in range(nrows)]
        })


        mask = np.isfinite(temp["VAR_orig"])
        temp.loc[mask, "VAR_fqc"] = 0

        if (not fill_all):
            temp["VAR_fall"] = temp["VAR_orig"]

        null_rows = temp["VAR_orig"].isna().sum()
        print(f'Initialized variable {var} with {null_rows} gaps to estimate uncertainities')


        self.sTEMP = self.sTEMP.merge(temp, left_index=True, right_index=True)

    
    def gap_filling_one_scenario(self, fluxVar, uStarTh, filterDayTime = False):

        ds = self.EddyData

        COLNAME = uStarTh.columns[1]
        
        if "season" not in ds.columns:
            ds["season"] = self.create_season_factor_month(ds["DateTime"])

        
        sTEMP = self.sTEMP

        uStarThresholds = pd.merge(sTEMP['season'], uStarTh, on = "season")[f'{COLNAME}']

        uStar = ds['Ustar']
        uStarq = np.zeros((ds.shape[0], 1), dtype=int)

        if filterDayTime:
            isFiltered = True 
        else:
            isFiltered = (~np.isfinite(ds['Rg'])) | (ds['Rg'] < 10)

        
        uStarq[(isFiltered) & (np.isfinite(uStarThresholds)) & (uStar < uStarThresholds)] = 1
        uStarq[(isFiltered) & (~np.isfinite(uStarThresholds))] = 3
        uStarq[(isFiltered) & (~np.isfinite(uStar))] = 4


        f_rows = (uStarq != 0).sum()

        print(f'Marked {round(100 * f_rows/uStarq.shape[0], 2)} of the data as gaps') 

        sTEMP[f'USTAR_{COLNAME}_thres'] = uStarThresholds
        sTEMP[f'USTAR_{COLNAME}_fqc'] = uStarq

        # print(self.sTEMP.head())
        # print("\n")

        # self.sTEMP_init(fluxVar, f'USTAR_{uStarTh.columns[1]}_fqc', 0)

        self.MDS_gap_fill(fluxVar, f'USTAR_{COLNAME}_fqc', 0)
        # self.sTEMP = sTEMP

    
    def MDS_gap_fill(self, fluxVar, qc_colname = "none", qc_val = 0, V1 = "Rg", T1 = 50, V2 = "VPD", T2 = 5, V3 = "Tair", T3 = 2.5, fill_all = True):
        
        self.TEMP_init(fluxVar, qc_colname, qc_val)
        ds = self.EddyData
        

        if (((V1 == "Rg") and (V1 not in ds.columns)) or (V1 == fluxVar)):      V1 = "none"
        if (((V2 == "VPD") and (V2 not in ds.columns)) or (V2 == fluxVar)):     V2 = "none"
        if (((V3 == "Tair") and (V3 not in ds.columns)) or (V3 == fluxVar)):    V3 = "none"

        # TODO: check that T1, T2, T3 columns are numeric and have non zero length

        if (V1 != "none" and V2 != "none" and V3 != "none" and 
            ds[V1].notna().sum() and ds[V2].notna().sum() and ds[V3].notna().sum()):
            
            print(f"Full MDS algorithm for gap filling of {fluxVar} with LUT({V1}, {V2},{V3}) and MDC")
            MET = 3 

        elif (V1 != "none" and ds[V1].notna().sum()):
            print(f"Limited MDS algorithm for gap filling of {fluxVar} with LUT({V1}) only and MDC")
            MET = 1

        else:
            print(f"Restricted MDS algorithm for gap filling of {fluxVar} with no conditions and MDC only")
            MET = 0

   
        if (MET == 3):
            for window in [7, 14]:
                self.fill_LUT(window, V1, T1, V2, T2, V3, T3)

        if (MET == 1 or MET == 3):
            self.fill_LUT(7, V1, T1)

        for window in [0, 1, 2]:
            self.fill_MDC(window)


        if (MET == 3):
            for window in range(21, 77, 7):
                self.fill_LUT(window, V1, T1, V2, T2, V3, T3)

        
        if (MET == 3 or MET == 1):
            for window in range(14, 77, 7):
                self.fill_LUT(window, V1, T1)

        for window in range(7, 217, 7):
            self.fill_MDC(window)

        # TODO: RESOLVE LONG GAPS TO NA OR SOMETHING?


        dsT = self.sTEMP
        print(f'\nFinished gap filling of variable of {fluxVar}')
        print(f'Artificial gaps filled: {len(dsT) - dsT["VAR_fall"].isna().sum()}, real gaps filled: {dsT["VAR_orig"].isna().sum()}, unfilled long gaps: {dsT["VAR_fall"].isna().sum()}\n')
        
        cols = [i for i in self.sTEMP.columns]     

        if (qc_colname != "none"):
            cols = [i.replace("VAR", fluxVar + "_" + qc_colname.split("_")[1]) for i in cols]
        else:
            cols = [i.replace("VAR", fluxVar) for i in cols]


        self.sTEMP.columns = cols


    # GAP FILLING ENTRY POINT
    def gap_fill(self, fluxVar, filterDayTime = False, annual_mapp = True):

        if (not annual_mapp):
            self.ustar_mapp = self.get_seasonal_map(self.ustar_results)

        if (not self.sTEMP):
            self.sTEMP = self.EddyData[['DateTime', 'season']].copy()

        
        thresholds = self.ustar_mapp

        for i in range(1, len(thresholds.columns)):
            self.gap_filling_one_scenario(fluxVar, thresholds[["season", thresholds.columns[i]]])
            # print(self.sTEMP.columns)


    def fill_LUT(self, window_size, V1 = "none", T1 = np.nan, V2 = "none", T2 = np.nan, V3 = "none", T3 = np.nan):
        
        dsT = self.sTEMP
        ds = self.EddyData

        lGF = pd.DataFrame(np.nan, index=[], columns=['index', 'mean', 'fnum', 'fsd', 'fmeth', 'fwin', 'fqc'])
        tofill = dsT[dsT['VAR_fall'].isna()].index.tolist()

        if len(tofill):
            cols = [i for i in [V1, V2, V3] if i != "none"]
            print(f"\nLook up table with window size {window_size} days with {cols}")

            for i in range(len(tofill)):        
                
                gap = tofill[i]
                
                # if (i % 100 == 0):
                #     print(".", end = "")
                # if (i % 2000 == 0):
                #     print()

                if (True):
                    start = gap - (window_size * self.DTS)
                    end = gap + (window_size * self.DTS)
                else:
                    start = gap - (window_size * self.DTS - 1)
                    end = gap + (window_size * self.DTS - 1)

                if (start < 0):                 start = 0
                if (end >= dsT.shape[0]):       end = dsT.shape[0] - 1 


                T1_mod = T1
                if "Rg" in V1:
                    T1_mod = max(min(T1, ds[V1][gap]), 20) 
                
                dt = {}
                for col in cols:
                    dt[col] = ds.loc[start:end, col]
                    dt[col] = dt[col].reset_index(drop = True)


                subgap = gap - (start)
                
                subset = dsT.loc[start:end, "VAR_orig"].reset_index(drop = True)
                rows_valid = np.isfinite(subset)
    
                for k, col in enumerate(cols):
                    values = dt[col]

                    if (col == "Rg"):
                        thres = T1_mod
                    elif (col == V1):
                        thres = T1 
                    elif (col == V2):
                        thres = T2 
                    else:
                        thres = T3
            
                    rows_valid = rows_valid & (np.isfinite(values)) & (abs(values - values[subgap]) < thres) 
                                             
                
                rows_LUT = dsT.loc[start:end, 'VAR_orig'].reset_index(drop=True)[rows_valid.values]
                v1_bel_t1 = dt[V1][rows_valid] <= dt[V1][subgap] 
                
                isNightTime = False
                if "Rg" in V1:
                    isNightTime = ds.loc[gap, V1] < 10 

                # if (len(rows_LUT) == 0):
                #     print(i)

                if (len(rows_LUT) > 1):
                    index = gap 
                    mean, fnum, fsd = rows_LUT.mean(), rows_LUT.shape[0], np.std(rows_LUT, ddof=1)

                    fwin = 2 * window_size
                    fmeth = fqc = np.nan

                    if (V1 != "none" and V2 != "none" and V3 != "none"):
                        fmeth = 1
                        if (fwin <= 14):                    fqc = 1
                        if (fwin > 14 and fwin <= 56):      fqc = 2
                        if (fwin > 56):                     fqc = 3

                    elif (V1 != "none"):
                        fmeth = 2
                        if (fwin <= 14):                    fqc = 1
                        if (fwin > 14 and fwin <= 28):      fqc = 2
                        if (fwin > 28):                     fqc = 3


                    lGF.loc[len(lGF)] = [index, mean, fnum, fsd, fmeth, fwin, fqc]


        if (len(tofill)):
            print(f'{lGF.shape[0]}')

        if (lGF.shape[0] > 0):
            
            dsT.loc[lGF['index'], ['VAR_fall', 'VAR_fnum', 'VAR_fsd', 'VAR_fmeth', 'VAR_fwin', 'VAR_fall_qc']] = \
                lGF[['mean', 'fnum', 'fsd', 'fmeth', 'fwin', 'fqc']].values
            
            gaps = dsT.loc[lGF['index'], 'VAR_f'].isna()
            gap_indices = lGF.index[gaps]
            dsT.loc[lGF.loc[gap_indices, 'index'], ['VAR_f', 'VAR_fqc']] = lGF.loc[gap_indices, ['mean', 'fqc']].values

    
    def fill_MDC(self, window_size):
        dsT = self.sTEMP
        ds = self.EddyData

        lGF = pd.DataFrame(np.nan, index=[], columns=['index', 'mean', 'fnum', 'fsd', 'fmeth', 'fwin', 'fqc'])
        tofill = dsT[dsT['VAR_fall'].isna()].index.tolist()

        if (len(tofill) > 1):
            print(f"\nMean diurnal course with window size {window_size} days")

            for i in range(len(tofill)):
                gap = tofill[i]
                window_indices = []

                for day in range(0, window_size + 1):
                    if day == 0:
                        window_indices.extend([gap + offset for offset in range(-2, 3)])
                    else:
                        window_indices.extend([
                            gap + (-day * self.DTS + offset) for offset in range(-2, 3)
                        ])
                        window_indices.extend([
                            gap + (day * self.DTS + offset) for offset in range(-2, 3)
                        ])

                window_indices = [idx for idx in window_indices if 0 <= idx < len(dsT)]

                rows_MDC = dsT.loc[window_indices, 'VAR_orig']
                rows_MDC = rows_MDC[rows_MDC.notna()]


                if (len(rows_MDC) > 1):
                    # print(rows_MDC)
                    index = gap
                    mean, fnum, fsd = rows_MDC.mean(), rows_MDC.shape[0], np.std(rows_MDC, ddof=1)
                    fmeth = 3

                    if (True or window_size < 7):
                        fwin = 2*window_size + 1
                    else:
                        fwin = window_size+1 


                    if (fwin <= 1):                     fqc = 1
                    elif (fwin > 1 and fwin <= 5):      fqc = 2
                    else:                               fqc = 3

                    lGF.loc[len(lGF)] = [index, mean, fnum, fsd, fmeth, fwin, fqc]


        if (len(tofill)):
            print(lGF.shape[0])

        if (lGF.shape[0] > 0):
            dsT.loc[lGF['index'], ['VAR_fall', 'VAR_fnum', 'VAR_fsd', 'VAR_fmeth', 'VAR_fwin', 'VAR_fall_qc']] = \
                lGF[['mean', 'fnum', 'fsd', 'fmeth', 'fwin', 'fqc']].values
            
            gaps = dsT.loc[lGF['index'], 'VAR_f'].isna()
            gap_indices = lGF.index[gaps]
            dsT.loc[lGF.loc[gap_indices, 'index'], ['VAR_f', 'VAR_fqc']] = lGF.loc[gap_indices, ['mean', 'fqc']].values

                    
                
    
















################# UTILS ###########################

def filter_long_runs_in_vector(x, min_run_length=8, replacement=np.nan, na_rm=True):
    """
    Replace runs of numerically equal values in a vector with a specified replacement value.

    Parameters:
        x (pd.Series or np.ndarray): The input vector to filter.
        min_run_length (int): Minimum length of a run to replace. Defaults to 8.
        replacement (any): Value to replace the original values in long runs. Defaults to NaN.
        na_rm (bool): If True, NA values are ignored when detecting runs. Defaults to True.

    Returns:
        pd.Series: The filtered vector with long runs replaced.
    """
    if na_rm:
        y = x.dropna().values if isinstance(x, pd.Series) else x[~np.isnan(x)]
    else:
        y = x

    if len(y) == 0:
        return x

    # Detect runs of equal values

    run_starts = np.where(np.diff(y, prepend=np.nan) != 0)[0]
    run_lengths = np.diff(np.append(run_starts, len(y)))

    # Replace values in long runs
    for start, length in zip(run_starts, run_lengths):
        if length >= min_run_length:
            y[start:start + length] = replacement

    # Reconstruct the original vector
    if na_rm:
        result = x.copy()
        result[~x.isna()] = y
        return result
    else:
        return pd.Series(y, index=x.index) if isinstance(x, pd.Series) else y

def filter_long_runs(data, col_names, min_run_length=8, replacement=np.nan, na_rm=True):
    """
    Replace runs of numerically equal values in specified columns of a DataFrame with a replacement value.

    Parameters:
        data (pd.DataFrame): The input DataFrame to filter.
        col_names (list): List of column names to filter.
        min_run_length (int): Minimum length of a run to replace. Defaults to 8.
        replacement (any): Value to replace the original values in long runs. Defaults to NaN.
        na_rm (bool): If True, NA values are ignored when detecting runs. Defaults to True.

    Returns:
        pd.DataFrame: The filtered DataFrame with long runs replaced in specified columns.
    """
    def apply_filter(column):
        return filter_long_runs_in_vector(column, min_run_length, replacement, na_rm)

    filtered_data = data.copy()
    for col in col_names:
        filtered_data[col] = apply_filter(filtered_data[col])

    return filtered_data

