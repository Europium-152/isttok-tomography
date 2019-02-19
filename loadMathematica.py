# -*- coding: utf-8 -*-
"""
Load Excel Files Exported by Mathematica
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp2d
# Assign spreadsheet filename to `file`
excel_file = 'los-three-test.xlsx'

# Load Spreadsheet. dfs is a dictonary of data frames corresponding to each sheet on the excel
dfs = pd.read_excel(excel_file,None,None)

exact_pro_3d=np.array([np.array(dfs[key]) for key in dfs])

x_file = 'x-axis.xlsx'
y_file = 'y-axis.xlsx'
z_file = 'z-axis.xlsx'

x_values = np.array(pd.read_excel(x_file,header=None),dtype="float64").flatten()
y_values = np.array(pd.read_excel(y_file,header=None),dtype="float64").flatten()
z_values = np.array(pd.read_excel(z_file,header=None),dtype="float64").flatten()
 
pro=interp2d(x_values,y_values,exact_pro_3d[0],bounds_error=False,fill_value=0.0)
    