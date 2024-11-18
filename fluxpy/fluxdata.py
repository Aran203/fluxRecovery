import pandas as pd
from .utils import INTERNAL_VAR_NAMES

class FluxData:

    def __init__(self, df: pd.DataFrame, site_elevation: float, site_lat: float, site_long: float, variable_map):
        ''' 
            Constructor with the following parameters
                df : dataframe object containing data
                site_elevation: elevation of site data is associated with
                site_lat: latitude of site
                site_long: longitude of site
                variable_map: dict that maps internal names to tuples of (user defined names, units)
        '''

        self.site_elevation = site_elevation
        self.site_latitude = site_lat
        self.site_longitude = site_long

        self.variable_map = {}

        df_l, map_, soil_vars_ = self.process_df_with_mapping(df, variable_map)

        self._df = df_l 
        self.variable_map.update(map_)
        self.soil_vars_weights = soil_vars_



    def process_variable_map(self, variable_map):
        vars_map = {}
        internal_var_names = set(INTERNAL_VAR_NAMES)

        if (isinstance(variable_map, list)):

            for element in variable_map:
                if len(element) != 2:
                    raise ValueError(f"List length isn't 2 for {element}")
                
                if (element[0] not in internal_var_names):
                    if not ((element[0].startswith('g_') or element[0].startswith('theta_'))):
                        raise ValueError(f'{element[0]} is not recogized as an internal name')
                
            for element in variable_map:
                vars_map[element[0]] = element[1]

        elif (isinstance(variable_map, dict)):
            keys_to_delete = []

            # delete any keys which map to empty strings
            for key in variable_map.keys():
                if isinstance(variable_map[key], str):
                    if len(variable_map[key]) == 0:
                        keys_to_delete.append(key)

            for key in keys_to_delete:
                del variable_map[key]

            for key in variable_map.keys():
                if key not in internal_var_names:
                    if not ((key.startswith('g_') or key.startswith('theta_'))):
                        raise ValueError(f'{key} is not recogized as an internal name')

                vars_map[key] = variable_map[key]
    
        return vars_map

    def process_soil_variables(self, variable_map):
        soil_vars = {key: value for key, value in variable_map.items() 
                     if (key.startswith('g_') or (key.startswith('theta_')))}

        soil_vars_map = {}

        for var in soil_vars.keys():
            soil_vars_map[var] = (soil_vars[var], 1)   # weight of 1 by default

        return soil_vars_map
        

    def process_df_with_mapping(self, df, variable_map):
        df = df.copy()
        var_map = self.process_variable_map(variable_map)
        soil_vars_map = self.process_soil_variables(var_map)

        input_cols_all = set(df.columns)

        # first only read columns which are present
        cols_present = []

        for key in var_map.keys():
            input_col_name = var_map[key]
            if input_col_name in input_cols_all:
                cols_present.append(input_col_name)
            else:
                print(f"Column {input_col_name} could not be found in dataset")
        
        df = df[cols_present]

        # now modify any input column names if they have the same name as internal names in both df and var_map
        internal_var_names = set(INTERNAL_VAR_NAMES)
        mod_dict = {}

        for key in var_map.keys():
            input_col_name = var_map[key]

            if (input_col_name in internal_var_names):
                mod_dict[input_col_name] = "INPUT_" + input_col_name
                var_map[key] = "INPUT_" + input_col_name


        for key in mod_dict:
            print(f"Renaming {key} to {mod_dict[key]}")
        
        df = df.rename(columns = mod_dict)

        # compute averages of soil variables if present
        df = self.compute_avg_soil_vars(df, soil_vars_map)


        return df, var_map, soil_vars_map
    
    
    def compute_avg_soil_vars(self, df, soil_vars_weights):
        ''' helper function to calculate the averages of soil variables (theta and g) based on weights '''
        g_mean = None
        theta_mean = None

        df = df.copy()

        g_vars = {key: value for key, value in soil_vars_weights.items() if key.startswith('g_')}
        g_mean = pd.Series([0] * len(df))
        g_weight_total = 0

        for key in g_vars.keys():
            column = g_vars[key][0]
            weight = g_vars[key][1]

            g_weight_total += weight 
            g_mean = g_mean + (df[column] * weight)

        if (g_weight_total != 0):
            print("Calculating mean for variable G")
            df['g_mean'] = g_mean / g_weight_total
            self.add_to_variable_map('g_mean', 'g_mean')

        theta_vars = {key : value for key, value in soil_vars_weights.items() if key.startswith('theta_')}
        theta_mean = pd.Series([0] * len(df))
        theta_weight_total = 0

        for key in theta_vars.keys():
            column = theta_vars[key][0]
            weight = theta_vars[key][1]

            theta_weight_total += weight
            theta_mean += (df[column] * weight)
        
        if (theta_weight_total != 0):
            print("Calculating mean for variable THETA")
            df['theta_mean'] = theta_mean / theta_weight_total
            self.add_to_variable_map('theta_mean', 'theta_mean')


        return df
    

    def add_to_variable_map(self, internal_name, user_name):
        ''' helper function to add to variable map afterwards; used when averages are computed
            overwrites values for previously defined internal names'''

        self.variable_map[internal_name] = user_name

        

    # Getters
    def get_latitude(self):
        return self.latitude
    
    def get_longitude(self):
        return self.longitude 
    
    def get_elevation(self):
        return self.elevation
