
from .fluxdata import FluxData
from .utils import GRIDMET_KEYS, AGG_DICT

import pandas as pd
import numpy as np
import xarray
from refet.calcs import _ra_daily, _rso_simple
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


class VeriFlux(FluxData):

    def __init__(self, flux_data: FluxData, drop_gaps = True, daily_frac = 1.00, max_interp_hours_day = 2, 
                 max_interp_hours_night = 4, gridMET_data = None):
        
        # getting site information from flux_data object
        self.site_elevation = flux_data.get_elevation()
        self.site_latitude = flux_data.get_latitude()
        self.site_longitude = flux_data.get_longitude()
        self.freq = flux_data.get_freq()

        self._df = flux_data._df
        self.daily_df = None
        self.variable_map = flux_data.variable_map
        self.inv_variable_map = {value : key for key, value in self.variable_map.items() if (value in self._df.columns)}

        self.flux_data = flux_data
        self.gridMET_data = gridMET_data

        self.daily_df = self.temporal_aggregation(drop_gaps, daily_frac, max_interp_hours_day, max_interp_hours_night)
        self.corrected_daily_df = None
        
        # print(self.daily_df[['INPUT_H', 'INPUT_LE', 'H_subday_gaps', 'LE_subday_gaps', 'G_subday_gaps', 'Rn_subday_gaps']])

    def add_to_variable_map(self, internal_name, user_name):
        ''' helper function to add to variable map afterwards'''

        self.variable_map[internal_name] = user_name
        self.inv_variable_map[user_name] = internal_name

    def temporal_aggregation(self, drop_gaps, daily_frac, max_interp_hours_day, max_interp_hours_night):

        # make copy and rename for internal usage
        df = self._df.copy().rename(columns = self.inv_variable_map)

        # freq = pd.infer_freq(df.index)

        # if freq and freq < 'D':
        #     print("Input data temporal frequency is less than daily")
        # elif freq and freq > 'D':
        #     print("Input data temporal frequency is more than daily")
        # elif (df.index.to_series().dt.normalize().is_unique):
        #     # If the day component is unique in indices, frequency has to be > than daily
        #     print("Input data temporal frequency seems to be daily")
        #     freq = 'D'

        #     idx = pd.date_range(df.index.min(), df.index.max())
        #     df = df.reindex(idx)
        #     df.index.name = 'date'
        # else:
        #     freq = None
        
        
        if self.freq == 'D':
            return self._df.rename(columns = self.variable_map)
        
        energy_vars = {'LE', 'H', 'Rn', 'G'}
        asce_std_vars = {'t_avg','sw_in','ws','vp'}
        
        energy_vars = energy_vars.intersection(df.columns)
        asce_std_vars = asce_std_vars.intersection(df.columns)
        interp_vars = list(asce_std_vars) + list(energy_vars)
        
        for col in energy_vars:
            df[f'{col}_subday_gaps'] = False
            df.loc[df[col].isna(), f'{col}_subday_gaps'] = True


        print('Data is being resampled to daily temporal frequency.')
        sum_cols = [k for k, v in AGG_DICT.items() if v == 'sum']
        sum_cols = list(set(sum_cols).intersection(df.columns))
        mean_cols = list(set(df.columns) - set(sum_cols))
                

        means = df[mean_cols].apply(pd.to_numeric, errors='coerce').resample('D').mean().copy()
        sums = df[sum_cols].dropna().apply(pd.to_numeric, errors='coerce').resample('D').sum()

        if max_interp_hours_day:
            freq_hrs = self.frequency_to_hours(self.freq)
            daily_measurements = 24 / freq_hrs


            day_gap = int(max_interp_hours_day / freq_hrs)
            night_gap = int(max_interp_hours_night / freq_hrs)

            tmp = df[interp_vars].apply(pd.to_numeric, errors='coerce')

            grped_night = tmp.loc[(tmp.Rn < 0) & (tmp.Rn.notna())].copy()
            grped_night.drop_duplicates(inplace=True)
            grped_night = grped_night.groupby(pd.Grouper(freq='24h', offset='12h'), group_keys=True).apply(
                            lambda x: x.interpolate(method='linear', limit=night_gap, limit_direction='both', limit_area='inside'))

            grped_day = tmp.loc[(tmp.Rn >= 0) | (tmp.Rn.isna())].copy()
            grped_day.drop_duplicates(inplace=True)
            grped_day = grped_day.groupby(pd.Grouper(freq='24h'),group_keys=True).apply(
                            lambda x: x.interpolate(method = 'linear', limit = day_gap, limit_direction='both', limit_area='inside'))
                

            if type(grped_night.index) is pd.MultiIndex: 
                grped_night.index = grped_night.index.get_level_values(1)
            

            if type(grped_day.index) is pd.MultiIndex:
                grped_day.index = grped_day.index.get_level_values(1)

            interped = pd.concat([grped_day, grped_night])
            if interped.index.duplicated().any():
                interped = interped.loc[~interped.index.duplicated(keep='first')]


            means[interp_vars] = interped[interp_vars].resample('D').mean().copy()


            if 't_avg' in interp_vars:
                means['t_min'] = interped.t_avg.resample('D').min()
                means['t_max'] = interped.t_avg.resample('D').max()
                self.variables['t_min'] = 't_min'
                self.units['t_min'] = self.units['t_avg']
                self.variables['t_max'] = 't_max'
                self.units['t_max'] = self.units['t_avg']
                interp_vars = interp_vars + ['t_min','t_max']
                interped[['t_min','t_max']] = means[['t_min','t_max']]

            if drop_gaps:

                n_vals_needed = int(24 / freq_hrs)

                data_cols = [ c for c in df.columns if not c.endswith('_qc_flag')]

                if max_interp_hours_day:
                    df[interp_vars] = interped[interp_vars].copy()
                    
                days_with_gaps = df[data_cols].groupby(df.index.date).count() < n_vals_needed

            df = means.join(sums)

            if drop_gaps:
                print(f'Filtering days with less then {daily_frac * 100}% sub-daily measurements')
                df[days_with_gaps] = np.nan

    
        df = df.rename(columns = self.variable_map)
        return df
            
    def frequency_to_hours(self, freq):
        unit_to_hours = {
            'T': 1 / 60,  # Minutes to hours
            'H': 1,       # Hours
            'D': 24,      # Days to hours
            'W': 24 * 7,  # Weeks to hours
            'M': 24 * 30, # Approximate (30 days)
            'Y': 24 * 365 # Approximate (365 days)
        }
        
        # Extract the numeric and unit parts
        num = ''.join(filter(str.isdigit, freq)) or '1' 
        unit = ''.join(filter(str.isalpha, freq))
        
        # Convert to hours
        if unit not in unit_to_hours:
            raise ValueError(f"Unsupported frequency unit: {unit}")
        
        return int(num) * unit_to_hours[unit]

    def _ET_gap_fill(self, et_name='ET_corr', refET='ETr'):
        
        df = self.corrected_daily_df.copy()
        

        _et_gap_fill_vars = ('ET_gap', 'ET_fill', 'ET_fill_val', 'ETrF', 'ETrF_filtered', 'EToF', 'EToF_filtered')

        vars = [i for i in _et_gap_fill_vars if i in set(df.columns)]
        if vars:
            df.drop(columns = vars, inplace = True)
        df = df.rename(columns = self.inv_variable_map)
        

        if not self.gridMET_data:
            self.gridMET_data = self.fetch_gridMET_data()
        
            
        vars_to_remove = [i for i in self.gridMET_data.columns if i in set(df.columns)]
        if vars_to_remove:
            self.gridMET_data.drop(columns = vars_to_remove, inplace = True)

        df = df.join(self.gridMET_data)
        for col in self.gridMET_data.columns:
            self.variable_map[col] = col

        if et_name not in df.columns:
            print(f'ERROR: {et_name} not found in data, cannot gap-fill')
            return

        print(f'Gap filling {et_name} with filtered {refET}F x {refET} (gridMET)')

        if refET == 'ETr':
            df['ETrF'] = df[et_name].astype(float) / df.gridMET_ETr.astype(float)
            df['ETrF_filtered'] = df['ETrF']
            # filter out extremes of ETrF
            Q1 = df['ETrF_filtered'].quantile(0.25)
            Q3 = df['ETrF_filtered'].quantile(0.75)
            IQR = Q3 - Q1
            to_filter = df.query('ETrF_filtered<(@Q1-1.5*@IQR) or ETrF_filtered>(@Q3+1.5*@IQR)')

            df.loc[to_filter.index, 'ETrF_filtered'] = np.nan
            df['ETrF_filtered'] = df.ETrF_filtered.rolling(7, min_periods=2, center=True).mean()

            df.ETrF_filtered = df.ETrF_filtered.interpolate(method='linear')
            df['ET_fill'] = df.gridMET_ETr * df.ETrF_filtered
            df['ET_gap'] = False
            df.loc[(df[et_name].isna() & df.ET_fill.notna()), 'ET_gap'] = True
            df.loc[df.ET_gap, et_name] = df.loc[df.ET_gap, 'ET_fill']

        elif refET == 'ETo':
            df['EToF'] = df[et_name].astype(float) / df.gridMET_ETo.astype(float)
            df['EToF_filtered'] = df['EToF']

            Q1 = df['EToF_filtered'].quantile(0.25)
            Q3 = df['EToF_filtered'].quantile(0.75)
            IQR = Q3 - Q1
            to_filter = df.query('EToF_filtered<(@Q1-1.5*@IQR) or EToF_filtered>(@Q3+1.5*@IQR)')
            df.loc[to_filter.index, 'EToF_filtered'] = np.nan
            df['EToF_filtered'] = df.EToF_filtered.rolling(7, min_periods=2, center=True).mean()
            df.EToF_filtered = df.EToF_filtered.interpolate(method='linear')

            # calc ET from EToF_filtered and ETo
            df['ET_fill'] = df.gridMET_ETo * df.EToF_filtered
            df['ET_gap'] = False

            df.loc[(df[et_name].isna() & df.ET_fill.notna()), 'ET_gap'] = True
            df.loc[df.ET_gap, et_name] = df.loc[df.ET_gap, 'ET_fill']
            

        if et_name == 'ET_corr' and 't_avg' in df.columns:
            df['LE_corr'] = (df.ET_corr * (2501000 - 2361 * df.t_avg.fillna(20))) / 86400
        
        elif et_name == 'ET_corr' and not 't_avg' in df.columns:
            df['LE_corr'] = (df.ET_corr * (2501000 - 2361 * 20)) / 86400

        df['ET_fill_val'] = np.nan
        df.loc[df.ET_gap , 'ET_fill_val'] = df.ET_fill

        new_cols = set(df.columns) - set(self.variable_map)
        for col in new_cols:
            self.variable_map[col] = col
        

        self.daily_df = df

    def _calc_rso(self):
        
        if 'rso' in self.daily_df.columns:
            return

        doy = self.daily_df.index.dayofyear
        latitude_rads = self.site_latitude * (np.pi / 180)

        ra_mj_m2 = _ra_daily(latitude_rads, doy, method='asce')
        rso_a_mj_m2 = _rso_simple(ra_mj_m2, self.site_elevation)

        self.daily_df['rso'] = rso_a_mj_m2 * 11.574
        self.variable_map.update({'rso' : 'rso'})

    def _calc_et(self):
        
        df = self.corrected_daily_df.copy()
        df = df.rename(columns = self.inv_variable_map)

        vars = ['ET', 'ET_corr', 'ET_user_corr']
        vars_to_drop = []

        for var in vars:
            if var in df.columns:
                vars_to_drop.append(var)
        
        if vars_to_drop:
            df.drop(columns = vars_to_drop, inplace = True)

        if not set(['LE','LE_corr','LE_user_corr']).intersection(df.columns):
            print('ERROR: no LE variables found in data')
            return
        
        if 't_avg' in df.columns:
            df['ET'] = 86400 * (df.LE / (2501000 - (2361 * df.t_avg.fillna(20))))
            if 'LE_corr' in df.columns:
                df['ET_corr'] = 86400 * (df.LE_corr / (2501000 - (2361 * df.t_avg.fillna(20))))
            if 'LE_user_corr' in df.columns:
                df['ET_user_corr'] = 86400*(df.LE_user_corr / (2501000 - (2361 * df.t_avg.fillna(20))))
        else:
            df['ET'] = 86400 * (df.LE / (2501000 - (2361 * 20)))
            if 'LE_corr' in df.columns:
                df['ET_corr'] = 86400 * (df.LE_corr/(2501000 - (2361 * 20)))
            if 'LE_user_corr' in df.columns:
                df['ET_user_corr']=86400*(df.LE_user_corr/(2501000-(2361*20)))
        
        new_cols = set(df.columns) - set(self.variable_map)
        for col in new_cols:
            self.variable_map[col] = col
        
        self.inv_variable_map = {value : key for key, value in self.variable_map.items() if (value in self._df.columns)}
        self.corrected_daily_df = df.rename(columns = self.variable_map)


    def _ebr_correction(self):
        wind_1 = 15
        wind_2 = 11
        half_win_1 = wind_1 // 2
        half_win_2 = wind_2 // 2

        df = self.daily_df.copy()

        _eb_calc_vars = ('br', 'br_user_corr', 'energy', 'energy_corr', 'ebr', 'ebr_corr', 'ebr_user_corr',
        'ebc_cf', 'ebr_5day_clim', 'flux', 'flux_corr', 'flux_user_corr','G_corr','H_corr', 'LE_corr', 'Rn_corr')

        vars = [i for i in _eb_calc_vars if i in set(self.daily_df.columns)]
        if vars:
            df.drop(columns = vars)

        df = df.rename(columns = self.inv_variable_map)

        vars_to_use = ['LE','H','Rn','G']

        potential_vars = {'SH', 'SLE', 'SG'}
        vars_to_use += [i for i in potential_vars if i in df.columns]


        df = df[vars_to_use].astype(float).copy()

        
        orig_df = df[['LE','H','Rn','G']].astype(float).copy()
        orig_df['ebr'] =  (orig_df.H + orig_df.LE) / (orig_df.Rn - orig_df.G)

        df['ebr'] = (df.H + df.LE) / (df.Rn - df.G)
        Q1 = df['ebr'].quantile(0.25)
        Q3 = df['ebr'].quantile(0.75)
        IQR = Q3 - Q1

        # filter values between Q1-1.5IQR and Q3+1.5IQR
        filtered = df.query('(@Q1 - 1.5 * @IQR) <= ebr <= (@Q3 + 1.5 * @IQR)')

        filtered_mask = filtered.index
        removed_mask = set(df.index) - set(filtered_mask)
        removed_mask = pd.to_datetime(list(removed_mask))
        df.loc[removed_mask] = np.nan

        ebr = df.ebr.values
        df['ebr_corr'] = np.nan

        for i in range(len(ebr)):
            win_arr1 = ebr[i-half_win_1:i+half_win_1+1]
            win_arr2 = ebr[i-half_win_2:i+half_win_2+1]
            count = np.count_nonzero(~np.isnan(win_arr1))
            # get median of daily window1 if half window2 or more days exist
            if count >= half_win_2:
                val = np.nanpercentile(win_arr1, 50, axis=None)
                if abs(1/val) >= 2 or abs(1/val) <= 0.5:
                    val = np.nan
            # if at least one day exists in window2 take mean
            elif np.count_nonzero(~np.isnan(win_arr2)) > 0:
                val = np.nanmedian(win_arr2)
                if abs(1/val) >= 2 or abs(1/val) <= 0.5:
                    val = np.nan
            else:
                # assign nan for now, update with 5 day climatology
                val = np.nan
            # assign values if they were found in methods 1 or 2
            df.iloc[i, df.columns.get_loc('ebr_corr')] = val


        
        doy_ebr_mean=df['ebr_corr'].groupby(df.index.dayofyear).mean().copy()
        l5days = pd.Series( index=np.arange(-4,1), data=doy_ebr_mean.iloc[-5:].values)
        f5days = pd.Series(index=np.arange(367,372), data=doy_ebr_mean.iloc[:5].values)
        
        #doy_ebr_mean = doy_ebr_mean.append(f5days)
        doy_ebr_mean = pd.concat([doy_ebr_mean, f5days])
        doy_ebr_mean = pd.concat([l5days, doy_ebr_mean])
        ebr_5day_clim = pd.DataFrame( index=np.arange(1,367), columns=['ebr_5day_clim'])
        doy_ebr_mean = doy_ebr_mean.values
        
        for i in range(len(doy_ebr_mean)):
            win = doy_ebr_mean[i:i+2*half_win_2+1]
            count = np.count_nonzero(~np.isnan(win))
            if i in ebr_5day_clim.index and count > 0:
                ebr_5day_clim.iloc[i-1, ebr_5day_clim.columns.get_loc('ebr_5day_clim')] = np.nanmean(win)
        
        ebr_5day_clim['DOY'] = ebr_5day_clim.index
        ebr_5day_clim.index.name = 'date'

        df['DOY'] = df.index.dayofyear

        null_dates = df.loc[df.ebr_corr.isnull(), 'ebr_corr'].index
        merged = pd.merge(df, ebr_5day_clim, left_on='DOY', right_index=True)
        merged.loc[null_dates, 'ebr_corr'] = merged.loc[null_dates, 'ebr_5day_clim'].astype(float)

        merged.LE = orig_df.LE
        merged.H = orig_df.H
        merged.Rn = orig_df.Rn
        merged.G = orig_df.G
        merged.ebr = orig_df.ebr

        merged['ebc_cf'] = 1/merged.ebr_corr
        merged.loc[ (abs(merged.ebc_cf) >= 2) | (abs(merged.ebc_cf <= 0.5)), 'ebc_cf'] = np.nan
        
        merged['LE_corr'] = merged.LE * merged.ebc_cf
        merged['H_corr'] = merged.H * merged.ebc_cf
       
        merged.loc[(merged.LE_corr >= 850) | (merged.LE_corr <= -100), ('LE_corr', 'H_corr', 'ebr_corr', 'ebc_cf')] = np.nan
        merged['flux_corr'] = merged['LE_corr'] + merged['H_corr']

        df = self.daily_df.rename(columns = self.inv_variable_map)

        df['flux'] = merged.LE + merged.H
        df['energy'] = merged.Rn - merged.G

        # corrected turbulent flux if given from input data
        if set(['LE_user_corr','H_user_corr']).issubset(df.columns):
            df['flux_user_corr'] = df.LE_user_corr + df.H_user_corr 
            df['ebr_user_corr']=(df.H_user_corr+df.LE_user_corr)/(df.Rn - df.G)
            df.ebr_user_corr=df.ebr_user_corr.replace([np.inf,-np.inf], np.nan)
            self.variable_map.update(flux_user_corr = 'flux_user_corr', ebr_user_corr = 'ebr_user_corr')
        
        cols = list(set(merged.columns).difference(df.columns))
        merged = df.join(merged[cols], how='outer')
        merged.drop('DOY', axis=1, inplace=True)

        self.variable_map.update(
            energy = 'energy',
            flux = 'flux',
            LE_corr = 'LE_corr',
            H_corr = 'H_corr',
            flux_corr = 'flux_corr',
            ebr = 'ebr',
            ebr_corr = 'ebr_corr',
            ebc_cf = 'ebc_cf',
            ebr_5day_clim = 'ebr_5day_clim'
        )

        self.corrected_daily_df = merged.rename(columns = self.variable_map)

    def _bowen_ratio_correction(self):
        
        df = self.daily_df.copy()

        _eb_calc_vars = ('br', 'br_user_corr', 'energy', 'energy_corr', 'ebr', 'ebr_corr', 'ebr_user_corr',
        'ebc_cf', 'ebr_5day_clim', 'flux', 'flux_corr', 'flux_user_corr','G_corr','H_corr', 'LE_corr', 'Rn_corr')

        vars = [i for i in _eb_calc_vars if i in set(self.daily_df.columns)]
        if vars:
            df.drop(columns = vars)

        df = df.rename(columns = self.inv_variable_map)

        df['br'] = df.H / df.LE
        df['LE_corr'] = (df.Rn - df.G) / (1 + df.br)
        df['H_corr'] = df.LE_corr * df.br
        df['flux_corr'] = df.LE_corr + df.H_corr

        
        if set(['LE_user_corr','H_user_corr']).issubset(df.columns):
            df['flux_user_corr'] = df.LE_user_corr + df.H_user_corr 
            df['br_user_corr'] = df.H_user_corr / df.LE_user_corr 
            df['ebr_user_corr']=(df.H_user_corr+df.LE_user_corr)/(df.Rn - df.G)
            df.ebr_user_corr=df.ebr_user_corr.replace([np.inf,-np.inf], np.nan)

            self.variable_map.update(
                flux_user_corr = 'flux_user_corr',
                ebr_user_corr = 'ebr_user_corr',
                br_user_corr = 'br_user_corr'
            )
        df['ebr'] = (df.H + df.LE) / (df.Rn - df.G)
        df['ebr_corr'] = (df.H_corr + df.LE_corr) / (df.Rn - df.G)
        df['energy'] = df.Rn - df.G
        df['flux'] = df.LE + df.H

        self.variable_map.update(
            br = 'br',
            energy = 'energy',
            flux = 'flux',
            LE_corr = 'LE_corr',
            H_corr = 'H_corr',
            flux_corr = 'flux_corr',
            ebr = 'ebr',
            ebr_corr = 'ebr_corr'
        )

        df.ebr = df.ebr.replace([np.inf, -np.inf], np.nan)
        df.ebr_corr = df.ebr_corr.replace([np.inf, -np.inf], np.nan)

        self.corrected_daily_df = df.rename(columns = self.variable_map)
    
    def _linear_regression(self, y, x, fit_intercept = False, apply_coefs = False):
        
        df = self.daily_df.copy()

        _eb_calc_vars = ('br', 'br_user_corr', 'energy', 'energy_corr', 'ebr', 'ebr_corr', 'ebr_user_corr',
        'ebc_cf', 'ebr_5day_clim', 'flux', 'flux_corr', 'flux_user_corr','G_corr','H_corr', 'LE_corr', 'Rn_corr')

        vars = [i for i in _eb_calc_vars if i in set(self.daily_df.columns)]
        if vars:
            df.drop(columns = vars)

        df = df.rename(columns = self.inv_variable_map)

        if not y in df.columns:
            print(f'ERROR: the dependent variable ({y}) was not found in the dataframe.')
            return
        if not isinstance(x, list) and not x in df.columns:
            print(f'ERROR: the dependent variable ({x}) was not found in the dataframe.')
            return

        n_x = 1
        if isinstance(x, list):
            n_x = len(x)
            if not set(x).issubset(df.columns):
                print('ERROR: one or more independent variables ({x}) were not found in the dataframe.')
                return
            if n_x > 1:
                cols = x + [y] 
                tmp = df[cols].copy()
        if n_x == 1:
            tmp = df[[x,y]].copy()

        tmp = tmp.dropna()
        X = tmp[x]
        Y = tmp[y]

        model = linear_model.LinearRegression(fit_intercept=fit_intercept)
        model.fit(X, Y)
        pred = model.predict(X)
        r2 = model.score(X,Y)
        rmse = (np.sqrt(mean_squared_error(Y, pred))).round(2)

        eb_vars = ['LE','H','Rn','G']
        if apply_coefs and set(tmp.columns).intersection(eb_vars):
            # calc initial energy balance if regression applied to EB vars
            df['ebr'] = (df.H + df.LE) / (df.Rn - df.G)
            df['flux'] = df.H + df.LE
            df['energy'] = df.Rn - df.G

        results = pd.Series()
        results.loc['Y (dependent var.)'] = y
        results.loc['c0 (intercept)'] = model.intercept_
        
        for i, c in enumerate(X.columns):
            results.loc['c{} (coef on {})'.format(i+1, c)] = model.coef_[i].round(3)
            if apply_coefs:
                new_var = '{}_corr'.format(c)
                # ensure correct sign of coef. if applied to EB vars
                if y in ['G','H','LE'] and c in ['G','H','LE']:
                    coef = -1 * model.coef_[i]
                else:
                    coef = model.coef_[i]
                print(f'Applying correction factor ({coef.round(3)}) to variable: {c} (renamed as {new_var}')

                df[new_var] = df[c] * coef
                self.variable_map[new_var] = new_var
                

        # results.loc['RMSE ({})'.format(self.units.get(y,'na'))] = rmse
        results.loc['r2 (coef. det.)'] = r2
        results.loc['n (sample count)'] = len(Y)
        results = results.to_frame().T
        # results.index= [self.site_id]
        results.index.name = 'SITE_ID'

        self.linear_regression_results = results

        if apply_coefs and set(X.columns).intersection(eb_vars):
            corr = pd.DataFrame()
            for v in eb_vars:
                cv = '{}_corr'.format(v)
                if cv in df:
                    corr[v] = df[cv] 
                else:
                    corr[v] = df[v]

            # not all vars are necessarily different than initial
            df['ebr_corr'] = (corr.H + corr.LE) / (corr.Rn - corr.G)
            df['flux_corr'] = corr.H + corr.LE
            df['energy_corr'] = corr.Rn + corr.G
            del corr

            self.variable_map.update(
                energy = 'energy',
                flux = 'flux',
                flux_corr = 'flux_corr',
                energy_corr = 'energy_corr',
                ebr = 'ebr',
                ebr_corr = 'ebr_corr'
            )

        self.corrected_daily_df = df.rename(columns = self.variable_map)

        return results

    def correct_data(self, meth = 'ebr', et_gap_fill = True, y = 'Rn', refET = 'ETr', x = ['G','LE','H'], fit_intercept = False):

        if not isinstance(self._df, pd.DataFrame):
            print('Please assign a dataframe of acceptable data first!')
            return
        if meth not in ['ebr', 'br', 'linear_regression']:
            err_msg = (f'ERROR: {meth} is not a valid correction option')
            raise ValueError(err_msg)

        self._calc_rso()

        eb_vars = {'Rn','LE','H','G'}
        if not eb_vars.issubset(self.variable_map.keys()) or \
            not eb_vars.issubset(self.daily_df.rename(columns=self.inv_variable_map).columns):

            print('Missing one or more energy balance variables, cannot perform energy balance correction.')
            self._calc_et()
            if et_gap_fill:
                self._ET_gap_fill(et_name='ET', refET=refET)
            return

        if meth == 'ebr':
            self._ebr_correction()
        elif meth == 'br':
            self._bowen_ratio_correction()
        else:
            self._linear_regression(y = y, x = x, fit_intercept = fit_intercept, apply_coefs = True)
        

        self._calc_et()
        if et_gap_fill:
            self._ET_gap_fill(et_name='ET_corr', refET = refET)

        self.inv_variable_map = {v: k for k, v in self.variable_map.items() if (
                not v.replace('_mean', '') == k or not k in self._df.columns)}
        
        if not 'G' in self.inv_variable_map.values():
            user_G_name = self.variable_map.get('G')
            self.inv_variable_map[user_G_name] = 'G'
    
    def set_gridMET_data(self, data):
        '''Sets gridMET data to be the data given
           data must be a pd.Dataframe or a csv        
        '''

        if (isinstance(data, pd.DataFrame)):
            self.gridMET_data = data
        else:
            self.gridMET_data = pd.read_csv(data)

    def fetch_gridMET_data(self):
        '''downloads gridMET data'''

        # opendap thredds server
        root = 'http://thredds.northwestknowledge.net:8080/thredds/dodsC/'

        variables = ['ETr', 'pet', 'pr']

            
        dates = self.daily_df.index
        gridmet_data_all = []
        print()

        for i, v in enumerate(variables):
            if v not in GRIDMET_KEYS:
                print(f'ERROR: {v} is not a valid gridMET variable')

            meta = GRIDMET_KEYS[v]

            self.add_to_variable_map(meta['rename'], meta['rename'])
            print(f'Downloading gridMET var: {meta["name"]}') 

            netcdf = f'{root}{meta["nc_suffix"]}'

            ds = xarray.open_dataset(netcdf).sel(lon = self.site_longitude, lat = self.site_latitude, method = 'nearest').drop('crs')
            df = ds.to_dataframe().loc[dates].rename(columns={meta['name']:meta['rename']})

            df.index.name = 'date' 

            df.drop(['lat', 'lon'], axis = 1, inplace = True)

            gridmet_data_all.append(df)
        
        # combine data
        df = pd.concat(gridmet_data_all, axis = 1)

        return df
        
    def save_gridMET_data(self, filename = None):
        '''Functionality to save fetched gridmet data in csv with specified filename if given '''

        if isinstance(self.gridMET_data, pd.DataFrame) and not self.gridMET_data.empty:
            print("Saving gridMET data")

            if not filename:
                self.gridMET_data.to_csv(f"gridMET_{self.site_latitude}_{self.site_longitude}.csv", index = False)
            else:
                self.gridMET_data.to_csv(filename, index = False)
        else:
            print("No gridMET data to save")