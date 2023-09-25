#! /usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

pd.options.mode.chained_assignment = None  # default='warn'


def func_lin(x, a, b):
  return a + b*x

parser = argparse.ArgumentParser()
parser.add_argument('-di', '--dir', help='input directory with mfd val', required=True)
parser.add_argument('-c', '--city_name', help='name of city', required=True)
args = parser.parse_args()

list_mode = ['class I','class II','class III','high-velocity']
marker = ['o','^','s','.']
#list_mode = ['walk','slow-mobility','car','high-velocity']

xlim_value=150

cname = args.city_name
list_fname=[]
for fname in os.listdir(args.dir):
  if fname.startswith(cname) and fname.endswith('mfd.csv'):
    list_fname.append(fname)

df_global=pd.DataFrame()
for fn in list_fname:
  input_file = os.path.join(args.dir, fn)
  print(input_file)
  df=pd.read_csv(input_file, sep=';')
  df["V_med"] = df["L_total"]/df["T_total"]
  df = df.fillna(0)
  list_speed = df['speed_av'].unique()
  list_idx = df['class'].unique()
  map_speed={}
  for i,j in zip(list_idx, list_speed):
    map_speed[i]=j
  map_speed=dict(sorted(map_speed.items(), key=lambda item: item[1]))
  print(map_speed, fn)
  map_speed_name = {}
  n=0
  for i,j in map_speed.items():
    map_speed_name[i]=list_mode[n]
    n+=1
  df['name_class'] = [map_speed_name[i] for i in df['class']]
  df_global = df_global.append(df, ignore_index=True)

list_x_value = np.arange(0,xlim_value,5)

dfgrp = df_global.groupby(by='name_class')
n=0
for idx, g in dfgrp:
  g.sort_values(by='density', inplace=True, ascending=False, ignore_index=True)
  g = g[g.density!=0]
  g['V_med_SMA'] = g.iloc[:,6].rolling(window=10).mean()
  g.dropna(inplace=True)
  modew=g.name_class.iloc[0]
  if modew=='high-velocity':
    n+=1
    continue
  plt.scatter(g['density'],g['V_med_SMA'],label=f'{modew}', s=7, marker=marker[n])
  popt_t, pcov = curve_fit(func_lin, g['density'].to_list(), g['V_med_SMA'].to_list())
  print(f'{idx} ---> {popt_t}')
  plt.plot(list_x_value, func_lin(list_x_value, *popt_t),'--', label='_nolegend_')
  #plt.plot(g['density'].to_list(), func_lin(g['density'], *popt_t),'--', label='_nolegend_')
  n+=1
plt.xlabel('N of activities')
plt.ylabel('V med (m/s)')
plt.xlim(0,xlim_value)
plt.legend()
#plt.show()
plt.savefig(f'mfd_{cname}_mdt.png', dpi=150)
plt.clf()
plt.close()
