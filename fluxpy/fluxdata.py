import pandas as pd

class FluxData:

    def __init__(self, df: pd.DataFrame, site_elevation: float, site_lat: float, site_long: float, variable_map):
        self._df = df
        self.site_elevation = site_elevation
        self.site_latitude = site_lat 
        self.site_longitude = site_long
        self.variable_map = self.process_variable_map(variable_map)


        # variable_names_dict = {
        #     'date' : 'datestring_col',
        #     'Rn' : 'net_radiation_col',
        #     'G' : 'ground_flux_col',
        #     'LE' : 'latent_heat_flux_col',
        #     'LE_user_corr' : 'latent_heat_flux_corrected_col',
        #     'H' : 'sensible_heat_flux_col',
        #     'H_user_corr' : 'sensible_heat_flux_corrected_col',
        #     'sw_in' : 'shortwave_in_col',
        #     'sw_out' : 'shortwave_out_col',
        #     'sw_pot' : 'shortwave_pot_col',
        #     'lw_in' : 'longwave_in_col',
        #     'lw_out' : 'longwave_out_col',
        #     'rh' : 'rel_humidity_col',
        #     'vp' : 'vap_press_col',
        #     'vpd' : 'vap_press_def_col',
        #     't_avg' : 'avg_temp_col',
        #     'ppt' : 'precip_col',
        #     'wd' : 'wind_dir_col',
        #     'ws' : 'wind_spd_col'
        # }


    def get_latitude(self):
        return self.latitude
    
    def get_longitude(self):
        return self.longitude 
    
    def get_elevation(self):
        return self.elevation
    
    def process_variable_map(self, variable_map):

        vars_map = {}

        if (isinstance(variable_map, list)):
            # if (len(variable_map) != len(self._df.columns)):
            #     raise ValueError

            # for element in variable_map:
            #     if len(element) != 3:
            #         raise ValueError
                
            for element in variable_map:
                vars_map[element[0]] = element[1]


        elif (isinstance(variable_map, dict)):

            for key in variable_map.keys():
                vars_map[key] = variable_map[key]


        return vars_map


                
            
    

