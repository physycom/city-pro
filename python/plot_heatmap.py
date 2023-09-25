import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import folium
from folium import plugins
from folium.plugins import HeatMap

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', help='presence input file', required=True)
  parser.add_argument('-p', '--place', help='Name of city', required=True)

  args = parser.parse_args()

  df = pd.read_csv(args.input, sep=';')
  df_count = df.groupby(by=['lat','lon']).size().reset_index(name='count')

  print(len(df_count))
  df_count=df_count[df_count['count'] > 20]
  print(len(df_count))

  map_hooray = folium.Map(location = [np.mean(df['lat'].to_numpy()),np.mean(df['lon'].to_numpy())],#location=[44.058763, 12.567305],
                    zoom_start = 13)

  # Ensure you're handing it floats
  #df_acc['Latitude'] = df_acc['Latitude'].astype(float)
  #df_acc['Longitude'] = df_acc['Longitude'].astype(float)

  # List comprehension to make out list of lists
  heat_data = [[row['lat'],row['lon']] for index, row in df_count.iterrows()]

  # Plot it on the map
  HeatMap(heat_data, radius=20, blur=20, max_zoom=7).add_to(map_hooray)

  # Display the map
  map_hooray.save('presence_{}.html'.format(args.place))
