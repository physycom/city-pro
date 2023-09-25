#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import argparse
import glob
import json
import os
import re
import sys
import pytz
from datetime import datetime
from collections import defaultdict
import folium
from matplotlib import cm
from matplotlib.colors import to_hex, rgb2hex

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cartogeo', help='carto geojson file', required=True)
  parser.add_argument('-f', '--fluxes', help='input city-pro timed_fluxes csv', required=True)
  args = parser.parse_args()

  cnt = pd.read_csv(args.fluxes, sep=';')
  cnt_total = cnt.groupby(['cid']).sum()
  cnt_total.drop(cnt_total.columns.difference(['total_fluxes']), 1, inplace=True)
  cnt_total = cnt_total*100

  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()

  cnt_total.hist(ax=ax, bins=70, rwidth=0.8)

  quantiles_val     = [0.0, 0.8, 0.85, 0.9, 0.95, 1.0]
  quantiles_val_col = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
  quantiles = cnt_total.total_fluxes.quantile(quantiles_val).to_list()
  bins = [int(q) for q in quantiles]

  cmap = cm.get_cmap('viridis', len(bins)-1)
  #cmap = cm.get_cmap('magma', len(bins)-1)
  #color_scale = [ to_hex(c) for c in cmap.colors ]
  #color_scale = color_scale[::-1]

  #cmap = cm.get_cmap('Reds', len(bins)-1)
  rgb_scale = cmap(quantiles_val_col)
  color_scale = [rgb2hex(r) for r in rgb_scale]
  color_scale = color_scale[:-1]
  map_col_quantil={}
  for i in np.arange(0, len(color_scale)):
    label = f'{quantiles_val[i]} < q <= {quantiles_val[i+1]}'
    map_col_quantil[color_scale[i]]=label

  ax.set_yscale('log')
  carto = gpd.read_file(args.cartogeo)
  list_color = pd.cut(cnt_total.total_fluxes, bins=bins, labels=color_scale).fillna(color_scale[0])
  carto.set_index('poly_cid', inplace=True)
  carto = carto.join(list_color)
  carto['quantile'] = [map_col_quantil[i] for i in carto.total_fluxes]

  carto = carto.iloc[1: , :]
  # layers
  center = carto.to_crs(epsg=3003).centroid.to_crs(epsg=4326)
  center = [ center.y.mean(), center.x.mean() ]

  # sort for color (for ordered visualization)
  carto.sort_values(by='quantile', inplace=True)

  m = folium.Map(location=center, control_scale=True)
  layerlabel = '<span style="color: {col};">{txt}</span>'
  for qval, grp in carto.groupby('quantile'):
    col = grp.total_fluxes.to_list()[0]
    flayer_sel = folium.FeatureGroup(name=layerlabel.format(col=col, txt=qval))
    for i, row in grp.iterrows():
      pol = folium.PolyLine(
        locations=[ [lat,lon] for lon,lat in row.geometry.coords],
        fill_color=col,
        color=col,
        #popup=folium.Popup(f'<p>name <b>{row.PRO_NAME}</b></br>id <b>{row.PRO}</b></p>', show=False, sticky=True),
      )
      flayer_sel.add_child(pol)
    m.add_child(flayer_sel)

  folium.map.LayerControl(collapsed=False).add_to(m)
  e, s, w, n = carto.total_bounds
  m.fit_bounds([ [s,w], [n,e] ])
  m.save(f'{args.fluxes}.html')

