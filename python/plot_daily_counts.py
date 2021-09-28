#! /usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import seaborn as sns

sns.set()


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input file of activity startstop', required=True)
args = parser.parse_args()

df = pd.read_csv(args.input, sep=';')
df['datetime_start']=[datetime.fromtimestamp(i) for i in df.timestart]
df['datetime_stop']=[datetime.fromtimestamp(i) for i in df.timestop]
dti = pd.date_range(start='15/08/2020', end='16/08/2020', freq="15T").to_pydatetime().tolist()
map_counting = {}
for i in dti:
  map_counting[i]=0

for index, row in df.iterrows():
  if index%1000==0:
    print(f'{index}/{len(df)}')
  for d in np.arange(0, len(dti)-1):
    if dti[d] <= row['datetime_start'] and dti[d+1] > row['datetime_start']:
      while row['datetime_stop'] > dti[d]:
        map_counting[dti[d]] += 1
        d+=1
      break

del map_counting[dti[-1]]
del dti[-1]
plt.plot(np.array(dti),np.array(list(map_counting.values())))
plt.xticks(rotation=45)
plt.xlabel('Hour (M-D H)')
plt.ylabel('N of activities in 15 min.')
plt.tight_layout()
plt.savefig('prova.png', dpi=300)
