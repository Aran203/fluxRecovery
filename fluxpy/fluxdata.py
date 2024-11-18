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

        df_l, map_ = self.process_df_with_mapping(df, variable_map)

        self._df = df_l 
        self.variable_map = map_



    def process_variable_map(self, variable_map):
        vars_map = {}
        internal_var_names = set(INTERNAL_VAR_NAMES)

        if (isinstance(variable_map, list)):

            for element in variable_map:
                if len(element) != 3:
                    raise ValueError(f"List length isn't 3 for {element}")
                
                # if (element[0] not in internal_var_names):
                #     raise ValueError(f'{element[0]} is not recogized as an internal name')
                
            for element in variable_map:
                vars_map[element[0]] = (element[1], element[2])

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
                # if key not in internal_var_names:
                #     raise ValueError(f'{key} is not recogized as an internal name')
                
                isTuple = isinstance(variable_map[key], tuple)
                
                if (not isTuple):
                    raise ValueError(f'Value for {key} is not a tuple')

                if (len(variable_map[key]) != 2):
                    raise ValueError(f'Tuple doesn\'t have length 2 for key {key}')

                vars_map[key] = variable_map[key]
    
        return vars_map


    def process_df_with_mapping(self, df, variable_map):
        df = df.copy()
        var_map = self.process_variable_map(variable_map)

        input_cols_all = set(df.columns)

        # first only read columns which are there
        cols_present = []

        for key in var_map.keys():
            input_col_name = var_map[key][0]
            if input_col_name in input_cols_all:
                cols_present.append(input_col_name)
            else:
                print(f"Column {input_col_name} could not be found in dataset")
        
        df = df[cols_present]

        # now modify any input column names if they have the same name as internal names in both df and var_map
        internal_var_names = set(INTERNAL_VAR_NAMES)
        mod_dict = {}

        for key in var_map.keys():
            input_col_name = var_map[key][0]

            if (input_col_name in internal_var_names):
                mod_dict[input_col_name] = "INPUT_" + input_col_name
                var_map[key] = ("INPUT_" + input_col_name, var_map[key][1])


        for key in mod_dict:
            print(f"Renaming {key} to {mod_dict[key]}")
        
        df = df.rename(columns = mod_dict)

        return df, var_map
    
    
    

    # Getters
    def get_latitude(self):
        return self.latitude
    
    def get_longitude(self):
        return self.longitude 
    
    def get_elevation(self):
        return self.elevation


                
            
    

