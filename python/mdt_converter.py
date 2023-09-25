# Import system modules
import json
import os
import argparse
import pandas as pd
import datetime 
import numpy as np
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
LIST_START = [datetime.datetime(2022,1,31,0,15,0),datetime.datetime(2022,2,11,0,30,0),
              datetime.datetime(2022,5,12,0,30,0),datetime.datetime(2022,7,1,0,14,0),
              datetime.datetime(2022,8,5,0,14,0),datetime.datetime(2022,11,11,0,14,0),
              datetime.datetime(2022,12,22,0,13,0),
              datetime.datetime(2022,12,30,0,13,0),
              datetime.datetime(2023,1,11,2,14,0),datetime.datetime(2023,2,1,2,14,0),
              datetime.datetime(2023,3,17,0,0,0)]

LIST_END = [datetime.datetime(2022,2,1,0,0,10),datetime.datetime(2022,2,12,0,0,27),
            datetime.datetime(2022,5,13,0,1,0),datetime.datetime(2022,7,2,0,1,0),
            datetime.datetime(2022,8,6,0,1,0),datetime.datetime(2022,11,12,0,0,19),
            datetime.datetime(2022,12,25,0,0,38),
            datetime.datetime(2023,1,2,0,0,59),
            datetime.datetime(2023,1,12,2,0,0),datetime.datetime(2023,2,2,2,0,0),
            datetime.datetime(2023,3,18,23,59,59)]


  # MICHELETTI
def sum_string(x,y):
    return [datetime.datetime.strptime(x.iloc[i] +' '+ y.iloc[i].split('.')[0],"%Y-%m-%d %H:%M:%S") for i in range(len(x))]

def city_pro_fit_data(df,small_filename,save_dir):
    df = df.sort_values('MTMSI')   
    mapper = {'Unnamed: 0': 'ID_TRIP','MTMSI':'CALL_ID','LATITUDE':'LAT','LONGITUDE':'LON','datetime':'timestamp'}
    df = df.rename(columns = mapper)
    tst = np.array(df['timestamp'])
    if type(tst[0]) == np.datetime64:
        df['timestamp'] = [(tst[i]- np.datetime64('1970-01-01T00:00:00Z'))/ np.timedelta64(1, 's') for i in range(len(tst))]
    else:
        df['timestamp'] = [datetime.datetime.strptime(tst[i],"%Y-%m-%d %H:%M:%S").timestamp() for i in range(len(tst))]     
    df = df[['CALL_ID','timestamp','LAT','LON']]
    df.to_csv(os.path.join(save_dir,small_filename),';', index = False)

def check_df_fill_emptyid(df):
    '''
    Solves the problem that in the data we have cases in which CALL_ID is \N but is in the middle of a trajectory.
    We assume it is in the same user.
    '''
    c=0
    wrong = []
    list_error = np.empty(0)
    i = 0
    for x in df.CALL_ID:
        try:
            a = np.longlong(x) 
        except ValueError:
            wrong.append(type(x))
            if df.CALL_ID.iloc[i-1]==df.CALL_ID.iloc[i+1] and isinstance(df.LAT.iloc[i],float) and isinstance(df.LON.iloc[i],float) and isinstance(df.timestamp.iloc[i],int):
                list_error= np.append(list_error,df.CALL_ID.iloc[i-1])
                if len(list_error)%1000==0:
                    print(len(list_error))
                    print(df.iloc[i])
            c+=1
        i+=1
    print(c)
    print(len(list_error))    

    
def evaluate_date_data(file_dir,chunk_size):
    global LIST_START
    global LIST_END
    big_file = pd.read_csv(file_dir, chunksize=chunk_size)
    start_date = datetime.datetime(2050,1,1,0,0,0)
    end_date = datetime.datetime(2000,1,1,0,0,0)
    for i, chunk in enumerate(big_file):
        chunk['datetime'] = sum_string(chunk['DATE'],chunk['TIME'])
        if min(chunk['datetime'])<=start_date:
            start_date = min(chunk['datetime'])
        else: 
            pass
        if max(chunk['datetime'])>=end_date:
            end_date = max(chunk['datetime'])
        else:
            pass
        for i in range(len(LIST_START)):
            if min(chunk['datetime'])>=LIST_START[i] and max(chunk['datetime'])<=LIST_END[i]:
                return LIST_START[i],LIST_END[i]
            else:
                pass 
    return start_date,end_date    
def generate_config_analysis(start_day,end_of_day,save_path,small_filename):
    global TIME_FORMAT    
    config_repo = '/home/aamad/codice/city-pro/work_geo/bologna_mdt_detailed/30-12'
    if os.path.exists(config_repo):
        with open(os.path.join(config_repo,'condif_parallel_bologna.json'),'r') as f:
            jsonfile = f.read()
        config_ = json.loads(jsonfile)
        config_['file_data'] = os.path.join(save_path,small_filename)
        config_['start_time'] = start_day.strftime(TIME_FORMAT)
        config_['end_time'] = end_of_day.strftime(TIME_FORMAT)
        config_['file_subnet'] = [] 
        config_['file_subnet'].append("/home/aamad/codice/city-pro/output/bologna_mdt_detailed/weights/{}.fluxes.sub".format(small_filename))
        default_repo = os.path.join("/home/aamad/codice/city-pro/work_geo/bologna_mdt_detailed",config_['start_time'].split(' ')[0])
        if not os.path.exists(default_repo):
            os.mkdir(default_repo)
        with open(os.path.join(default_repo,"config_bologna.json"), "w") as outfile:
            json.dump(config_, outfile,indent=4)
    return os.path.join(default_repo,"config_bologna.json") +' '


def cut_tdm_files(parameters):
    '''Input:
    work_dir: path to big file
    filename: name of big file
    save_dir: path to save smaller files
    num_chunks: 
    chunk_size: number of rows of uploaded piece of dataframe
    
    ----------------------------------
    Description:
    1) Takes a big file of activities of telecom data.
    2) Chooses time window to study and create csv file in that window
    3) Saves it as a file named bologna_mdt_{start_date}_{end_date}.csv
    '''
    file_dir = parameters[0]
    save_dir = parameters[1]
    chunk_size = parameters[2]
    start_date,end_date = evaluate_date_data(file_dir,chunk_size)
    if not os.path.isfile(os.path.join(save_dir,'considered_dates.txt')):            
        with open(os.path.join(save_dir,'considered_dates.txt'),'w') as f:
            f.write('start date:\t'+ start_date.strftime(TIME_FORMAT)+', end date:\t'+end_date.strftime(TIME_FORMAT)+'\n')
        f.close()
    else:
        with open(os.path.join(save_dir,'considered_dates.txt'),'a') as f:
            f.write('start date:\t'+ start_date.strftime(TIME_FORMAT)+', end date:\t'+end_date.strftime(TIME_FORMAT)+'\n')
        f.close()
        
#    print('start date:\n',start_date)
#    print('end date:\n',end_date)
    big_file = pd.read_csv(file_dir, chunksize=chunk_size)
    list_columns_to_save = ['MTMSI', 'LATITUDE', 'LONGITUDE','datetime']
    new_ = True
    for i, chunk in enumerate(big_file):
        if i == 0 or new_:            
            chunk['datetime'] = sum_string(chunk['DATE'],chunk['TIME'])
            chunk = chunk.sort_values(by = 'datetime')
            new = chunk[list_columns_to_save].loc[chunk['datetime']>=start_date]
            new = new[list_columns_to_save].loc[new['datetime']<=end_date]
            new_ = False
        else:
            chunk['datetime'] = sum_string(chunk['DATE'],chunk['TIME'])
            chunk = chunk.sort_values(by = 'datetime')
            df_tmp = chunk[list_columns_to_save].loc[chunk['datetime']>=start_date]
            df_tmp = df_tmp[list_columns_to_save].loc[df_tmp['datetime']<=end_date]
            new = pd.concat([new,df_tmp],ignore_index = True,verify_integrity = True)
    new = new.sort_values(by = 'datetime')   
#    print('chunk united together starting:\t',new.iloc[0]['datetime'],' ending:\t',new.iloc[-1]['datetime']) 
    delta = end_date - start_date
    if delta.days == 0:
        
        end_of_day = start_date.replace(hour=23, minute=59, second=59)
        start_day = start_date.replace(hour=0, minute=0, second=0)
#        print('dealing with day:\n',start_day,' of size: ',len(new))
        str_start_day = start_day.strftime(TIME_FORMAT)
        str_end_day = end_of_day.strftime(TIME_FORMAT)
        small_filename = 'bologna_mdt_{0}_{1}.csv'.format(str_start_day.split(' ')[0],str_end_day.split(' ')[0])
        save_path = os.path.join(save_dir,str_start_day.split(' ')[0])
        if not os.path.exists(save_path):
            os.mkdir(save_path) 
        if not os.path.isfile(os.path.join(save_path,small_filename)):            
            city_pro_fit_data(new,small_filename,save_path)
        else:
            pass
        input_file_next_step_analysis = generate_config_analysis(start_day,end_of_day,save_path,small_filename)
        print(input_file_next_step_analysis)
        with open(os.path.join(save_dir,'considered_dates.txt'),'a') as f:
            f.write('concluded analysis for start day:\t'+ str(start_day)+', end day:\t'+str(end_of_day)+'\n')
        f.close()
        with open(os.path.join(save_dir,'output_program.txt'),'a') as f:
            f.write(input_file_next_step_analysis)
        f.close()
        
    else:
        end_of_day = start_date.replace(hour=23, minute=59, second=59)
        start_day = start_date.replace(hour=0, minute=0, second=0)
        while(end_of_day<end_date):    
#            print('inside multiple days processing')    
            str_start_day = start_day.strftime(TIME_FORMAT)
            str_end_day = end_of_day.strftime(TIME_FORMAT)
#            print('\tdealing with day:\n\t',start_day,' ',end_of_day,' of size: ',len(new))

            small_filename = 'bologna_mdt_{0}_{1}.csv'.format(str_start_day.split(' ')[0],str_end_day.split(' ')[0])
            save_path = os.path.join(save_dir,str_start_day.split(' ')[0])
            df_tmp = new[list_columns_to_save].loc[new['datetime']>=start_day]
            df_tmp = df_tmp[list_columns_to_save].loc[df_tmp['datetime']<=end_of_day]
#            print('\tdealing with cutted day:\n\t',start_day,' ',end_of_day,' of size: ',len(df_tmp))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if not os.path.isfile(os.path.join(save_path,small_filename)):                                    
                city_pro_fit_data(df_tmp,small_filename,save_path)
            else:                
                pass
            input_file_next_step_analysis = generate_config_analysis(start_day,end_of_day,save_path,small_filename)
            with open(os.path.join(save_dir,'considered_dates.txt'),'a') as f:
                f.write('concluded analysis for start day:\t'+ str(start_day)+', end day:\t'+str(end_of_day)+'\n')
            f.close()
            with open(os.path.join(save_dir,'output_program.txt'),'a') as f:
                f.write(input_file_next_step_analysis)
            f.close()            
            print(input_file_next_step_analysis)
                        
            start_day = end_of_day +datetime.timedelta(seconds=1)
            end_of_day = end_of_day +datetime.timedelta(days=1)        
    new_ = True
    return True
        



if __name__ == '__main__':
    # PARSE FILE
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', help='directory of configuration file', required=True)
    args = parser.parse_args()
    with open(args.conf,'r') as f:
         jsonfile = f.read()
    config_ = json.loads(jsonfile)
    list_files = []
    for file in config_['list_files']:
#        print(os.path.exists(os.path.join(config_['base_dir'],file)))
        os.path.join(config_['base_dir'],file)
        list_files.append(os.path.join(config_['base_dir'],file))
    try:
        if len(config_['list_files1'])>0:
            for file1 in config_['list_files1']:
                list_files.append(os.path.join(config_['base_dir1'],file1))
    except:
        pass
    save_dir = config_['save_dir']
    chunk_size = 100000
    tuple_parameters = []
    for file_dir in list_files:
        tuple_parameters.append([file_dir,save_dir,chunk_size])
#    print(np.shape(tuple_parameters))
    # RUN
    with Pool() as p:
#        for result in p.map(cut_tdm_files,tuple_parameters):    
        p.map(cut_tdm_files,tuple_parameters)
#            print('DONE')
            
