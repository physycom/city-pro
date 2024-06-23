# CONTROLS THE CONSISTENCY OF THE DATASET
from multiprocessing import Pool
import pandas as pd
import os
import numpy as np
import datetime
import time
dates = ["2023-03-18","2022-01-31","2023-01-11","2023-02-01","2022-02-11","2022-05-12","2022-07-01","2022-08-05","2022-11-11","2022-12-30",
         "2022-12-31","2023-01-01"]
addresses = ["/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/{0}/bologna_mdt_{1}_{2}.csv".format(dates[0],dates[0],dates[0]),             
            "/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/{0}/bologna_mdt_{1}_{2}.csv".format(dates[1],dates[1],dates[1]),
            "/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/{0}/bologna_mdt_{1}_{2}.csv".format(dates[2],dates[2],dates[2]),
             "/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/{0}/bologna_mdt_{1}_{2}.csv".format(dates[3],dates[3],dates[3]),
             "/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/{0}/bologna_mdt_{1}_{2}.csv".format(dates[4],dates[4],dates[4]),
             "/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/{0}/bologna_mdt_{1}_{2}.csv".format(dates[5],dates[5],dates[5]),
             "/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/{0}/bologna_mdt_{1}_{2}.csv".format(dates[6],dates[6],dates[6]),
             "/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/{0}/bologna_mdt_{1}_{2}.csv".format(dates[7],dates[7],dates[7]),
             "/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/{0}/bologna_mdt_{1}_{2}.csv".format(dates[8],dates[8],dates[8]),
             "/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/{0}/bologna_mdt_{1}_{2}.csv".format(dates[9],dates[9],dates[9]),
             "/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/{0}/bologna_mdt_{1}_{2}.csv".format(dates[10],dates[10],dates[10]),
             "/home/aamad/codice/city-pro_old/work_geo/bologna_mdt_detailed/data/{0}/bologna_mdt_{1}_{2}.csv".format(dates[11],dates[11],dates[11])]

def check(address):
    print("access: ", address) 
    df = pd.read_csv(address,';')
    initial_size = len(df)
    is_longlong_func = lambda s: isinstance(s, np.int64) or (isinstance(s, str) and s.isdigit())
    mask = []
    start = time.time()
    # Iterate over the DataFrame rows
    for i, row in df.iterrows():
        CALL_ID = row['CALL_ID']

        # Check if the CALL_ID is not a longlong integer
        if not is_longlong_func(CALL_ID):
            # Check the previous row
            if i > 0 and is_longlong_func(df.at[i-1, 'CALL_ID']) and df.at[i-1, 'CALL_ID'] == df.at[i+1, 'CALL_ID']:
                df.at[i, 'CALL_ID'] = df.at[i-1, 'CALL_ID']
                mask.append(True)
                print('nan:\t',i,' ',address)
            # Check the next row
            elif i < len(df) - 1 and is_longlong_func(df.at[i+1, 'CALL_ID']) and df.at[i-1, 'CALL_ID'] == df.at[i+1, 'CALL_ID']:
                df.at[i, 'CALL_ID'] = df.at[i+1, 'CALL_ID']
                mask.append(True)
            elif is_longlong_func(df.at[i+1, 'CALL_ID']) and df.at[i-1, 'CALL_ID'] != df.at[i+1, 'CALL_ID']:
                mask.append(False)
                print('eliminated:\t',i,' ',address)
        else:
            mask.append(True)
        if i%100000==0:
            print('\t',address.split('/')[-1], ' time passed',round(time.time()-start,3),' iter: ',int(i/100000))
    df.to_csv(address.split('.')[0] +'_1.csv',';', index = False)
            
    df = df.loc[mask]
    print('fraction to initial size:\t',len(df)/initial_size)
    return True
def check1(address):
    print("access: ", address) 
    df = pd.read_csv(address,';')
    initial_size = len(df)
    start = time.time()
    i = 0
    mask = []
    for x in df.CALL_ID:
        try:
            a = np.longlong(x)
            mask.append(True) 
        except ValueError:
            if i<initial_size-1 and df.CALL_ID.iloc[i-1]==df.CALL_ID.iloc[i+1] and isinstance(df.LAT.iloc[i],float) and isinstance(df.LON.iloc[i],float) and isinstance(df.timestamp.iloc[i],int):
                df.CALL_ID.iloc[i] = df.CALL_ID[i-1] 
                mask.append(True)
            else:
                mask.append(False)
        i+=1
        if i%100000==0:
            print('\t',address.split('/')[-1], ' time passed',round(time.time()-start,3),' iter: ',int(i/100000))
    df =df.loc[mask]
    df.to_csv(address.split('.')[0] +'_1.csv',';', index = False)
                
        
def check_df_fill_emptyid(address):
    print("access: ", address) 
    df = pd.read_csv(address,';')
    print(df.columns)   
    df = df[-1000000:]
#    check_error_func = lambda i:np.isnan(np.longlong(df['CALL_ID'].iloc[i]))
    check_error_func = lambda i: isinstance(np.longlong(df['CALL_ID'].iloc[i]),np.int64) 
    
    check_double_error = lambda i:np.isnan(np.longlong(df['CALL_ID'].iloc[i])) and np.isnan(np.longlong(df['CALL_ID'].iloc[i-1]) and np.isnan(np.longlong(df['CALL_ID'].iloc[i+1])))
    filter_func = lambda i: (df['CALL_ID'].iloc[i-1] == df['CALL_ID'].iloc[i+1]) and \
                            isinstance(df['LAT'].iloc[i], float) and \
                            isinstance(df['LON'].iloc[i], float) and \
                            isinstance(df['timestamp'].iloc[i], int)

    # Get the indices where np.longlong(df.CALL_ID.iloc[i]) does not produce an error
#    mask = [True if check_error_func(i) else False for i in df.index.to_series()]
    nan_indices = df.index.to_series()[df.index.to_series().apply(check_error_func)]
    print('operation:\n',df.index.to_series()[mask])
    print("number of nan indeces ",len(nan_indices),' ',address)
    nan_indices_double = df.index.to_series()[df.index.to_series().apply(check_double_error)]
    nan_indices = [x for x in nan_indices if (not x in nan_indices_double)]
    print("number of nan indeces after comparison ",len(nan_indices),' ',address)
    # Apply filter_func and func_substitute on the valid indices
    c =0
    for i in nan_indices:
        if filter_func(i):
            c+=1
            df.at[i, 'CALL_ID'] = df.at[i-1, 'CALL_ID']
#    df.to_csv(address,';', index = False)
            print('\t',address,' ',c/len(df))
with Pool() as p:
#    p.map(check_df_fill_emptyid,addresses)
    p.map(check1,addresses)
    
    

