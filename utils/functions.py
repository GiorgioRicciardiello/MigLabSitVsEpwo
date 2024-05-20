import pandas as pd
import numpy as np

def osa_categories(ahi):
    """Categorize AHI value into OSA severity levels."""
    ahi_range = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
    if pd.isna(ahi):
        return np.nan
    elif ahi < 5:
        return ahi_range['Normal']
    elif 5 <= ahi < 15:
        return ahi_range['Mild']
    elif 15 <= ahi < 30:
        return ahi_range['Moderate']
    else:
        return ahi_range['Severe']
