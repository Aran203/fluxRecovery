
INTERNAL_VAR_NAMES = ['date', 'Rn', 'G', 'LE', 'LE_user_corr', 'H', 'H_user_corr', 'sw_in', 'sw_out', 'sw_pot', 'lw_in', 
                      'lw_out', 'rh', 'vp', 'vpd', 't_avg', 'ppt', 'wd', 'ws']

AGG_DICT = {
    'ASCE_ETo': 'sum',
    'ASCE_ETr': 'sum',
    'energy': 'mean',
    'flux': 'mean',
    'flux_corr': 'mean',
    'br': 'mean',
    'ET': 'sum',
    'ET_corr': 'sum',
    'ET_gap': 'sum',
    'ET_fill': 'sum',
    'ET_fill_val': 'sum',
    'ET_user_corr': 'sum',
    'ebr': 'mean',
    'ebr_corr': 'mean',
    'ebr_user_corr': 'mean',
    'ebr_5day_clim': 'mean',
    'gridMET_ETr': 'sum',
    'gridMET_ETo': 'sum',
    'gridMET_prcp': 'sum',
    'lw_in': 'mean',
    't_avg': 'mean',
    't_max': 'mean',
    't_min': 'mean',
    't_dew': 'mean',
    'rso': 'mean',
    'sw_pot': 'mean',
    'sw_in': 'mean',
    'vp': 'mean',
    'vpd': 'mean',
    'ppt': 'sum',
    'ppt_corr': 'sum',
    'ws': 'mean',
    'Rn': 'mean',
    'Rn_subday_gaps': 'sum',
    'rh' : 'mean',
    'sw_out': 'mean',
    'lw_out': 'mean',
    'G': 'mean',
    'G_subday_gaps': 'sum',
    'LE': 'mean',
    'LE_corr': 'mean',
    'LE_subday_gaps': 'sum',
    'LE_user_corr': 'mean',
    'H': 'mean',
    'H_corr': 'mean',
    'H_subday_gaps': 'sum',
    'H_user_corr': 'mean',
}

GRIDMET_KEYS = {
        'ETr': {
            'nc_suffix': 'agg_met_etr_1979_CurrentYear_CONUS.nc#fillmismatch',
            'name': 'daily_mean_reference_evapotranspiration_alfalfa',
            'rename': 'gridMET_ETr',
            'units': 'mm'
        },
        'pr': {
            'nc_suffix': 'agg_met_pr_1979_CurrentYear_CONUS.nc#fillmismatch',
            'name': 'precipitation_amount',
            'rename': 'gridMET_prcp',
            'units': 'mm'
        },    
        'pet': {
            'nc_suffix': 'agg_met_pet_1979_CurrentYear_CONUS.nc#fillmismatch',
            'name': 'daily_mean_reference_evapotranspiration_grass',
            'rename': 'gridMET_ETo',
            'units': 'mm'
        },
        'sph': {
            'nc_suffix': 'agg_met_sph_1979_CurrentYear_CONUS.nc#fillmismatch',
            'name': 'daily_mean_specific_humidity',
            'rename': 'gridMET_q',
            'units': 'kg/kg'
        },
        'srad': {
            'nc_suffix': 'agg_met_srad_1979_CurrentYear_CONUS.nc#fillmismatch',
            'name': 'daily_mean_shortwave_radiation_at_surface',
            'rename': 'gridMET_srad',
            'units': 'w/m2'
        },
        'vs': {
            'nc_suffix': 'agg_met_vs_1979_CurrentYear_CONUS.nc#fillmismatch',
            'name': 'daily_mean_wind_speed',
            'rename': 'gridMET_u10',
            'units': 'm/s'
        },
        'tmmx': {
            'nc_suffix': 'agg_met_tmmx_1979_CurrentYear_CONUS.nc#fillmismatch',
            'name': 'daily_maximum_temperature',
            'rename': 'gridMET_tmax',
            'units': 'K'
        },
        'tmmn': {
            'nc_suffix': 'agg_met_tmmn_1979_CurrentYear_CONUS.nc#fillmismatch',
            'name': 'daily_minimum_temperature',
            'rename': 'gridMET_tmin',
            'units': 'K'
        },
}