import pandas as pd

def fill_nulldata(df):
    for col in df.columns:
        
        if df[col].dtype == object:
            df.loc[df[col].isnull() == True, col] = df[col].mode()[0]# [0] is written for the object type data
            
        else:
            df.loc[df[col].isnull() == True, col] = df[col].mean()