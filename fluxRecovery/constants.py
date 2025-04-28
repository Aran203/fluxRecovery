
CORR_METHODS = (
    'ebr',
    'br',
    'lin_regress'
)

GRIDMET_METADATA = {
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

# all potentially calculated variables for energy balance corrections
_eb_calc_vars = (
    'br',
    'br_user_corr',
    'energy',
    'energy_corr',
    'ebr',
    'ebr_corr',
    'ebr_user_corr',
    'ebc_cf',
    'ebr_5day_clim',
    'flux',
    'flux_corr',
    'flux_user_corr',
    'G_corr',
    'H_corr',
    'LE_corr',
    'Rn_corr'
)
# potentially calculated variables for ET
_et_calc_vars = (
    'ET',
    'ET_corr',
    'ET_user_corr'
)
# potentially calculated ET gap fill variables
_et_gap_fill_vars = (
    'ET_gap',
    'ET_fill',
    'ET_fill_val',
    'ETrF',
    'ETrF_filtered',
    'EToF',
    'EToF_filtered'
)