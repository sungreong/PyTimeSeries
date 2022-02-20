from this import d
import pandas as pd

def make_lower_column(df:pd.DataFrame):
    new_cols = [i.lower() for i in list(df)]
    df.columns = new_cols 
    return df 

def make_upper_column(df:pd.DataFrame):
    new_cols = [i.upper() for i in list(df)]
    df.columns = new_cols 
    return df 


