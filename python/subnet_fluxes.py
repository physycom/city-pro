#! /usr/bin/env python

import os
import glob
import folium
import argparse
import numpy as np
import geopandas as gpd
from datetime import datetime

def classes(date):
  files = sorted(glob.glob(output_folder + '/*class_*.txt'))
  files_subnet = sorted(glob.glob(output_folder + '/*class_subnet*.txt'))
  files_class = [item for item in files if item not in files_subnet]

  dict_subnet = {}
  for file_class in files_class:
    tok = file_class[:file_class.find('.')].split('_')[-1]
    values = np.loadtxt(file_class, dtype='int')
    dict_subnet[tok] = values.tolist()
  return dict_subnet

def color_map(val):
  color_dict = {
    1.0: '#FF0000',   # Red
    2.0: '#808080',   # Gray
    3.0: '#0000FF',   # Blue
    4.0: '#FFFF00',   # Yellow
  }
  return color_dict.get(val, '')

def subnet_map(carto_df, output_folder):
  center = carto_df.to_crs(epsg=3003).centroid.to_crs(epsg=4326)
  center = [ center.y.mean(), center.x.mean() ]
  m = folium.Map(location=center, control_scale=True)
  import warnings
  warnings.filterwarnings("ignore")#, category=ShapelyDeprecationWarning)
  layerlabel = '<span style="color: {col};">{txt}</span>'
  for s_net, grp in carto_df.groupby('sub_net'):
    col = grp.color.to_list()[0]
    if col == '#000000': flayer_sel = folium.FeatureGroup(name=layerlabel.format(col=col, txt=f'Cluster NAN'))
    else: flayer_sel = folium.FeatureGroup(name=layerlabel.format(col=col, txt=f'Cluster {int(s_net)}'))
    for i, row in grp.iterrows():
      pol = folium.PolyLine(
        locations=[ [lat,lon] for lon,lat in row.geometry.coords],
        fill_color=col,
        color=col,
        popup=folium.Popup(f'<p>Cluster: <b>{round(row.sub_net)}', show=False, sticky=True),
      )
      flayer_sel.add_child(pol)
    m.add_child(flayer_sel)
  folium.map.LayerControl(collapsed=True).add_to(m)
  e, s, w, n = carto_df.total_bounds
  m.fit_bounds([ [s,w], [n,e] ])
  file_name = f'{date.strftime("%Y%m%d")}_subclasses_map'
  html_file = f'{output_folder}/{file_name}.html'
  m.save(html_file)
  print(f'\nFile saved in {os.path.abspath(html_file)} ')

def valid_date(s):
  try:
    return datetime.strptime(s, "%Y-%m-%d")
  except ValueError:
    msg = "Not a valid date: '{0}'.".format(s)
    raise argparse.ArgumentTypeError(msg)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--date', help='input date', required=True, type=valid_date)
  args = parser.parse_args()

  carto_file =  os.path.join(os.environ['WORKSPACE'], 'city-pro', 'vars', 'extra', 'bologna-provincia.geojson')
  date = args.date.date()

  work_base = os.path.join(os.environ['WORKSPACE'], 'city-pro', 'work_bologna')
  if not os.path.exists(work_base): os.mkdir(work_base)
  output_folder = os.path.join(os.environ['WORKSPACE'], 'city-pro', 'work_bologna', 'output', f'{date.strftime("%Y-%m-%d")}')
  if not os.path.exists(output_folder): os.mkdir(output_folder)

  class_list = classes(date.strftime('%Y-%m-%d'))
  dict_poly_subnet = {}
  for k, v in class_list.items():
    for item in v:
      dict_poly_subnet[item] = int(k)

  carto_df = gpd.read_file(carto_file)
  carto_df['sub_net'] = carto_df['poly_lid'].map(dict_poly_subnet)

  carto_df['color'] = carto_df['sub_net'].apply(color_map)
  carto_df.loc[carto_df['color'] == '', 'color'] = '#000000'
  print_map = subnet_map(carto_df, output_folder)