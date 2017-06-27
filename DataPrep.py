import pandas as pd
import numpy as np

def GlassImport():
    # import dataset
    cols = ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'type']
    glass = pd.read_csv('glass.txt', names=cols)
    # create window target column (1 for true; 0 for false)
    glass.ix[glass.ix[:, 'type'] <= 4, 'window'] = 1
    glass.ix[glass.ix[:, 'type'] > 4, 'window'] = 0
    # create non-window target column (1 for true; 0 for false)
    glass.ix[glass.ix[:, 'type'] > 4, 'non_window'] = 1
    glass.ix[glass.ix[:, 'type'] <= 4, 'non_window'] = 0
    # drop id and class columns
    glass.drop(['id', 'type'], inplace=True, axis=1)
    return glass
