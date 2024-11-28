
from .fluxdata import FluxData
from .utils import GRIDMET_KEYS, AGG_DICT
import pandas as pd
import numpy as np

class VeriFlux(FluxData):

    def __init__(self, flux_data: FluxData, drop_gaps = True, daily_frac = 1.00, max_interp_hours_day = 2, 
                 max_interp_hours_night = 4, gridMET_data = None):
        
        # getting site information from flux_data object
        self.site_elevation = flux_data.get_elevation()
        self.site_latitude = flux_data.get_latitude()
        self.site_longitude = flux_data.get_longitude()
        self.freq = flux_data.get_freq()

        self._df = flux_data._df
        self.monthly_df = None
        self.variable_map = flux_data.variable_map
        self.inv_variable_map = {value : key for key, value in self.variable_map.items() if (value in self._df.columns)}

        self.flux_data = flux_data
        self.gridMET_data = gridMET_data

        self.monthly_df = self.temporal_aggregation(drop_gaps, daily_frac, max_interp_hours_day, max_interp_hours_night)
        
        print(self.monthly_df[['INPUT_H', 'INPUT_LE', 'H_subday_gaps', 'LE_subday_gaps', 'G_subday_gaps', 'Rn_subday_gaps']])
         

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
            print("DF")
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

        # print(sum_cols)
        # print(mean_cols)
        
        

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

            print(means)

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
                print(n_vals_needed)

                data_cols = [ c for c in df.columns if not c.endswith('_qc_flag')]

                if max_interp_hours_day:
                    df[interp_vars] = interped[interp_vars].copy()
                    
                days_with_gaps = df[data_cols].groupby(df.index.date).count() < n_vals_needed

            df = means.join(sums)

            if drop_gaps:
                print(f'Filtering days with less then {daily_frac * 100}% sub-daily measurements')
                print(days_with_gaps)
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

    def set_gridMET_data(self, data):
        '''Sets gridMET data to be the data given
           data must be a pd.Dataframe or a csv        
        '''

        if (isinstance(data, pd.DataFrame)):
            self.gridMET_data = data
        else:
            self.gridMET_data = pd.read_csv(data)

    def get_gridMET_data(self, data):
        '''downloads gridMET data'''
        # TODO

        pass

