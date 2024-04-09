import os
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import timedelta
from shapely.geometry import LineString, Point, Polygon
from datetime import datetime
import movingpandas as mpd
from shapely import wkt
from shapely.ops import nearest_points, unary_union
from collections import defaultdict
import multiprocessing
from tqdm import tqdm

from progressbar import Bar, ETA, Percentage, ProgressBar, ReverseBar

import warnings
warnings.simplefilter("ignore")


def geodata(file_path, create_json):
  # data_path = os.path.join(os.environ['WORKSPACE'], 'city-pro', 'input', 'data', 'viasat', 'viasat_200723', 'raw_data_gps','Forlì-Cesena_202002')
  # file_name = data_path + f'/VEM_FORLI_FEBBRAIO_GPSRAW_PRIVATEVEHICLES.csv'

  data_raw = pd.read_csv(file_path, sep = ";")
  data_raw[['LAT', 'LON']] = data_raw[['LAT', 'LON']].apply(lambda x: x.str.replace(',', '.').astype(float))
  if create_json:
    bbox = [data_raw['LAT'].max(), data_raw['LON'].max(), data_raw['LAT'].min(), data_raw['LON'].min()]
    data = {
            "osm_data" : "map.osm",
            "out_base" : "mymap",     

            "bbox": {
                "lat_max": bbox[0],
                "lat_min": bbox[2],
                "lon_max": bbox[1],
                "lon_min": bbox[3]
            }
    }
    with open("carto_conf.json", "w") as outfile:
      json.dump(data, outfile,indent=2)

  data_raw['DATA_REGISTRAZIONE'] = pd.to_datetime(data_raw['DATA_REGISTRAZIONE'], format='%d/%m/%Y %H:%M:%S')
  geometry = [Point(xy) for xy in zip(data_raw['LON'], data_raw['LAT'])]
  gdf = gpd.GeoDataFrame(data_raw, geometry=geometry, crs='epsg:4326')
  gdf.sort_values(by=['ID_VEICOLO', 'DATA_REGISTRAZIONE'], inplace=True)
  return gdf

def update_traj_id(row):
  if pd.to_timedelta(row['delta_t']) > timedelta(hours=1):
    update_traj_id.traj_counter +=1
  return row['traj_id'] + update_traj_id.traj_counter


def traj(gdf):
  if test: 
    start_time =  pd.to_datetime('2020-02-10 08:00:00')
    stop_time =  pd.to_datetime('2020-02-10 08:30:00')
    print(f'Start datetime test: {start_time}')
    print(f'Stop datetime test: {stop_time}\n')

    df = gdf.loc[(gdf['DATA_REGISTRAZIONE'] > start_time) & (gdf['DATA_REGISTRAZIONE'] <= stop_time)]
  else : 
    df = gdf
  df = df[['ID_VEICOLO', 'DATA_REGISTRAZIONE', 'geometry']].sort_values('DATA_REGISTRAZIONE')

  traj_df = pd.DataFrame(columns=['ID_VEICOLO', 'geometry', 'traj_id'])

  df['traj_id'] = 0
  for id_car, df_car in df.groupby('ID_VEICOLO'):
    df_car['tvalue'] = df_car['DATA_REGISTRAZIONE']
    df_car['delta_t'] = (df_car['tvalue']-df_car['tvalue'].shift()).fillna(0)
    update_traj_id.traj_counter = 0
    df_car['traj_id'] = df_car.apply(update_traj_id, axis=1)
    df_car = df_car.drop(columns=['tvalue','delta_t'])
    traj_df = pd.concat([traj_df, df_car], axis=0)

  df_sorted = traj_df.sort_index()
  widgets = [Bar('>'), ' ', ETA(), ' ', ReverseBar('<'), ' ', Percentage()]
  print(f'Start collect trajectories\n')
  pbar = ProgressBar(widgets=widgets, maxval=len(df_sorted.groupby('ID_VEICOLO'))).start()

  trajectories_df = pd.DataFrame()
  for bar_cnt, (id, df_id) in enumerate(df_sorted.groupby('ID_VEICOLO')):
    df_id = df_id.reset_index()
    # df_id['geometry'] = gpd.GeoSeries.from_wkt(df_id['geometry'])
    try:
      geo_df = gpd.GeoDataFrame(df_id, crs='epsg:4326', geometry = 'geometry')
      if len(geo_df) < 2 : continue 
      traj_collection = mpd.TrajectoryCollection(geo_df, 'traj_id', t='DATA_REGISTRAZIONE')
      traj_collection.add_distance(overwrite=False, name='distance', units=("km"))

      traj_gdf = traj_collection.to_traj_gdf(wkt=False)

      traj_gdf['id'] = id
      traj_gdf['start_t'] = pd.to_datetime(traj_gdf['start_t'])
      traj_gdf['end_t'] = pd.to_datetime(traj_gdf['end_t'])

      traj_gdf['delta_t'] = (traj_gdf['end_t'] - traj_gdf['start_t']).dt.total_seconds()
      traj_gdf['speed'] = (traj_gdf['length']) / traj_gdf['delta_t']
      traj_gdf = traj_gdf[['id','traj_id','start_t', 'end_t', 'geometry','length','delta_t','speed']]
      trajectories_df = pd.concat([trajectories_df, traj_gdf], axis=0)
      pbar.update(bar_cnt)
    except Exception as e:
      # print(f'Exception at {id} as {e}')
      pass 
 
  pbar.finish()
  trajectories_df.reset_index(drop=True, inplace=True)
  return trajectories_df

def OD_matrix(gdf, ace_shapefile, freq):
  for index, row in gdf.iterrows():
    coords = [(coords) for coords in list(row['geometry'].coords)]
    first_coord, last_coord = [ coords[i] for i in (0, -1) ]
    gdf.at[index,'start_pnt'] = Point(first_coord)
    gdf.at[index,'end_point'] = Point(last_coord)
  
  shape_polygon = gpd.read_file(ace_shapefile)
  start_points = gpd.GeoDataFrame(geometry=gdf['start_pnt'])
  end_points = gpd.GeoDataFrame(geometry=gdf['end_point'])

  start_points = gpd.sjoin(start_points, shape_polygon, how='left', op='within')
  end_points = gpd.sjoin(end_points, shape_polygon, how='left', op='within')
  gdf['ACE_orig'] = start_points['UID']
  gdf['ACE_dest'] = end_points['UID']

  df = gdf[['id', 'traj_id', 'start_t', 'end_t', 'ACE_orig', 'ACE_dest']]
  df['start_t'] = pd.to_datetime(df['start_t'])
  df['end_t'] = pd.to_datetime(df['end_t'])
  df['rounded_start_t'] = df['start_t'].sub(pd.Timedelta('1ns')).dt.round(freq=freq)
  df['rounded_end_t'] = df['end_t'].sub(pd.Timedelta('1ns')).dt.round(freq=freq)
  od_matrix = df.groupby([df['rounded_start_t'], df['rounded_end_t'], 'ACE_orig', 'ACE_dest']).size().reset_index(name='counter')
  od_matrix.columns = ['date', 'hour', 'ace_orig', 'ace_dest', 'counter']
  od_matrix['date_time'] = pd.to_datetime(od_matrix['date'])
  od_matrix.drop(columns=['date', 'hour'], inplace=True)

  od_matrix = od_matrix[['date_time', 'ace_orig', 'ace_dest', 'counter']]
  if save_output:
    od_save_path = f'{output_folder}/viasat_{first_date}_{last_date}_OD.csv'
    od_matrix.to_csv(od_save_path, sep=";")
    print(f'OD matrix saved in {os.path.abspath(od_save_path)}')
  else:
    print(f'OD matrix not saved')
    print(od_matrix)

  return od_matrix


def tessellation(bounding_coord, N_boxes):
  n = int(N_boxes)
  minx, miny, maxx, maxy = bounding_coord
  h_step = (maxx - minx) / n
  v_step = (maxy - miny) / n
  squares = []
  for i in range(n):
    for j in range(n):
      square_coords = [
          (minx + i * h_step, miny + j * v_step),
          (minx + (i + 1) * h_step, miny + j * v_step),
          (minx + (i + 1) * h_step, miny + (j + 1) * v_step),
          (minx + i * h_step, miny + (j + 1) * v_step)
      ]
      squares.append(Polygon(square_coords))

  square_grid = gpd.GeoDataFrame({'geometry': squares})
  length = np.sqrt(gpd.GeoSeries(square_grid['geometry'].values[0], crs="EPSG:4326").to_crs('epsg:3857').area / 10**6).values[0]
  return square_grid, length

def process_traj(traj_index, traj_row, small_boxes, carto_gdf):
  box4check = [box["geometry"] for _, box in small_boxes.iterrows() if any(box["geometry"].contains(Point(point)) for point in traj_row["geometry"].coords)]
  box4check = pd.DataFrame(box4check, columns=['geometry'])
  traj_linestring = traj_row['geometry']
  polygons_shape = gpd.GeoDataFrame(box4check, geometry='geometry')
  filtered_carto = gpd.sjoin(carto_gdf, polygons_shape, how='inner', op='within')
  filtered_carto = filtered_carto.drop(columns=['index_right'])

  poly_lids = []
  for point in traj_linestring.coords:
    point = Point(point)
    point_buffer = point.buffer(0.00015)
    for _, carto_row in filtered_carto.iterrows():
      carto_linestring = carto_row['geometry']
      if point_buffer.intersects(carto_linestring):
        poly_lids.append(carto_row['poly_lid'])
        break
  return {traj_index: list(set(poly_lids))}

  # poly_lids = []
  # for point in traj_linestring.coords:
  #   point = Point(point)
  #   point_buffer = point.buffer(0.00015)
  #   if point_buffer.intersects(carto_union):
  #     for carto_index, carto_row in carto_gdf.iterrows():
  #       carto_linestring = carto_row['geometry']
  #       if point_buffer.intersects(carto_linestring):
  #         poly_lids.append(carto_row['poly_lid'])
  #         break
  # return {traj_index: list(set(poly_lids))}
  

def match_on_carto(traj_df, carto, save_output, output_folder):
  carto_gdf = gpd.read_file(carto)
  bb_box = carto_gdf.geometry.total_bounds

  small_boxes, box_length = tessellation(bb_box, 50)
  if test:
    import folium
    m = folium.Map(location=[small_boxes.geometry[0].centroid.y, small_boxes.geometry[0].centroid.x], zoom_start=10)
    for _, row in small_boxes.iterrows():
      polygon_geojson = row['geometry'].__geo_interface__
      folium.GeoJson(polygon_geojson).add_to(m)

    m.save('polygons_test_map.html')

  map_traj = {}

  # widgets = [Bar('>'), ' ', ETA(), ' ', ReverseBar('<'), ' ', Percentage()]
  print(f'\nStart trajectories reconstruction\n- Time: {datetime.now()}')
  # pbar = ProgressBar(widgets=widgets, maxval=len(traj_df)).start()

  num_processes = multiprocessing.cpu_count()
  pool = multiprocessing.Pool(processes=num_processes)

  results = []
  for bar_index, (traj_index, traj_row) in enumerate(tqdm(traj_df.iterrows(), total=len(traj_df), desc='Processing')):
    results.append(pool.apply_async(process_traj, args=(traj_index, traj_row, small_boxes, carto_gdf)))
    # pbar.update(bar_index)

  for res in results:
    result = res.get()
    map_traj.update(result)

  pool.close()
  pool.join()
  # pbar.finish()

  traj_df['poly_lid'] = traj_df.index.map(map_traj)

  print(f'\End trajectories reconstruction\n- Time: {datetime.now()}\n')


  total_traj = len(traj_df)
  traj_loss = len(traj_df[traj_df['poly_lid'].apply(len) == 0])
  pct = (1- traj_loss/total_traj) * 100
  print(f'Loss data: \n- Total data: {total_traj}\n- Loss Data: {traj_loss}\n- Percentage of success: {pct:.2f}%')

  # if save_output:
  #   traj_save_path = f'{output_folder}/viasat_{first_date}_{last_date}_trajectories_not_ordered.csv'
  #   traj_df.to_csv(traj_save_path, sep=';')
  #   print(f'Trajectories saved in {os.path.abspath(traj_save_path)}')
  # else:
  #   print(f'Trajectories not saved\n')
  #   print(traj_df)

def get_first_last_points(line):
  first_point = Point(line.coords[0])
  last_point = Point(line.coords[-1])
  return first_point, last_point

def get_nearest_point(point, line):
  distances = [point.distance(p) for p in line]
  nearest_index = np.argmin(distances)
  return line[nearest_index], distances[nearest_index]

def sort_trajectory(traj_df, carto_file):
  carto_geo = gpd.read_file(carto_file)

  lid2geometry = defaultdict(lambda : {})
  for p_lid, geometry in carto_geo[['poly_lid', 'geometry']].values:
    lid2geometry[p_lid] = geometry

  def sort_function(lista, starting_point):
    line = []
    bound2poly = defaultdict(lambda: {})
    bound2poly_reverse = defaultdict(lambda: {})
    for p in lista:
      start_poly = get_first_last_points(lid2geometry[p])[0]
      end_poly   = get_first_last_points(lid2geometry[p])[1]
      bound2poly[start_poly][end_poly] = p
      bound2poly_reverse[end_poly][start_poly] = p
      line.append(start_poly)
      line.append(end_poly)

    nearest_point, distance = get_nearest_point(starting_point, line)

    if not bound2poly[nearest_point]:
      value = list(bound2poly_reverse[nearest_point].values())[0]
      second_point = list(bound2poly_reverse[nearest_point].keys())[0]
    else:
      value = list(bound2poly[nearest_point].values())[0]
      second_point = list(bound2poly[nearest_point].keys())[0]

    sorted_list.append(value)
    new_list = list(set(lista) - set(sorted_list))
    if len(new_list) != 0: 
      sort_function(new_list, second_point)
    return sorted_list

  for index, row in traj_df.iterrows():
    if len(row['poly_lid']) == 1:
      traj_df.at[index, 'poly_lid'] = [row['poly_lid'][0]]
    elif len(row['poly_lid']) != 0:
      row['start_pnt'] = str(get_first_last_points(row['geometry'])[0])
      sorted_list = []
      sort_function_case = sort_function(row.poly_lid, wkt.loads(row.start_pnt))
      traj_df.at[index, 'poly_lid'] = sort_function_case

  if save_output:
    traj_save_path = f'{output_folder}/viasat_{first_date}_{last_date}_trajectories.csv'
    traj_df.to_csv(traj_save_path, sep=';')
    print(f'Trajectories saved in {os.path.abspath(traj_save_path)}')
  else:
    print(f'Trajectories not saved\n')
    print(traj_df)


if __name__ == '__main__':
  import argparse
  
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help=f'config file')
  args = parser.parse_args()

  with open(args.cfg) as cfgfile:
    config = json.load(cfgfile)

  test = True

  data_path     = config["data_path"]
  ace_shapefile = config["ace_shapefile"]
  carto_conf    = config["carto_conf"]
  save_output   = config['save_output']
  output_folder = config["output_folder"]
  freq          = config["od_freq"]
  carto         = config["carto_geojson"]

  print(f'Starting analysis with \n- Test: {test}\n- Data Path: {data_path}\n- Shapefile Path: {ace_shapefile} \n- Save Output: {save_output}\n- Output Folder: {output_folder}\n')

  if not os.path.exists(output_folder): 
    try:
      os.makedirs(output_folder)
      print(f'\nCreated output path {output_folder}')
    except Exception as e:
      print(f'\nSave folder not created, check {e}')

  geo_df = geodata(data_path, False)
  first_date = str(geo_df['DATA_REGISTRAZIONE'].min()).replace('-','')[:8]
  last_date = str(geo_df['DATA_REGISTRAZIONE'].max()).replace('-','')[:8]


  trajectories_df = traj(geo_df)

  create_OD = OD_matrix(trajectories_df, ace_shapefile, freq)

  match_carto = match_on_carto(trajectories_df, carto, save_output, output_folder)

  sort_gdf = sort_trajectory(trajectories_df, carto)