# Import system modules
import sys
import os
import argparse
from os import listdir
from os.path import isfile, join
import glob
import pandas as pd
from datetime import datetime

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-fdir', '--file_dir', help='directory of chunk', required=True)
  parser.add_argument('-m', '--mode', choices=['mr','ue'], help='specify kind of data', required=True)
  parser.add_argument('-d', '--date', help='string date in format Y-m-d', required=True)
  args = parser.parse_args()

  if args.mode == 'mr':
    list_chunk =  glob.glob(f'{args.file_dir}/*_MR*.cpcsv')

  elif args.mode =='ue':
    list_chunk =  glob.glob(f'{args.file_dir}/*_UEINFORES*.cpcsv')

  df_glob = pd.DataFrame()
  for i in list_chunk:
    print(f"I'm reading {i} file")
    dfw = pd.read_csv(i, ';')
    dfw['date'] = [str(datetime.fromtimestamp(i).date()) for i in dfw.timestamp]
    dfw = dfw[dfw.date == args.date]
    df_glob = df_glob.append(dfw)

  df_glob = df_glob.drop(['date'], axis=1)
  df_glob.to_csv(f'Bologna_{args.date}_{args.mode}.csv', sep=';', index=False)