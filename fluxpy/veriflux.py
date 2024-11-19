
from .fluxdata import FluxData
from .utils import GRIDMET_KEYS
import pandas as pd

class VeriFlux(FluxData):

    def __init__(self, flux_data: FluxData, drop_gaps = True, daily_frac = 1.00, max_interp_hours_day = 2, 
                 max_interp_hours_night = 4, gridMET_data = None):
        
        self.flux_data = flux_data
        self.gridMET_data = gridMET_data
        pass


    
    

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
        pass

